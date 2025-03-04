import os
import argparse
import yaml
import aniposelib
import cv2
import numpy as np
import pandas as pd

from contextlib import contextmanager
@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()

def do_calibration(args):
    ## Run the calibration if necessary
    calib_file = os.path.join(args.base_path, args.calibration_file)

    if args.force_calibration or not os.path.exists(calib_file):
        if args.verbose > 0:
            if not os.path.exists(calib_file):
                print(f"No calibration file found. Running calibration")
            else:
                print("force_calibration is True. Running calibration")

        board = aniposelib.boards.CharucoBoard(squaresX=args.nsquaresx,
                                            squaresY=args.nsquaresy,
                                            square_length=args.square_length,
                                            marker_length=args.marker_length,
                                            marker_bits=args.marker_bits,
                                            dict_size=args.dict_size)

        assert len(args.calibration_videos) == len(args.camera_names), \
            f'Number of calibration videos {len(args.calibration_videos)} is different than number of camera names {len(args.camera_names)}'
        camgroup = aniposelib.cameras.CameraGroup.from_names(args.camera_names)

        vidnames = [[os.path.join(args.base_path, vid)] for vid in args.calibration_videos]
        err, rows = camgroup.calibrate_videos(vidnames, board, 
                                init_intrinsics=True, init_extrinsics=True, 
                                verbose=args.verbose > 0)
        
        if args.verbose == 2:
            # save the detected points
            df_all = []
            for cam1, rows_cam in zip(args.camera_names, rows):
                df_cam = []
                col_idx = pd.MultiIndex.from_product([[cam1], ['x', 'y']],
                                                    names = ['camera', 'coordinate'])
                
                for pts1 in rows_cam:
                    fr1 = pts1['framenum'][1]
                    row_idx = pd.MultiIndex.from_product([[fr1], pts1['ids'][:,0]], 
                                                        names=['frame', 'id'])
                    df_cam.append(pd.DataFrame(data = np.squeeze(pts1['corners']), 
                                            index=row_idx, 
                                            columns=col_idx))
                df_all.append(pd.concat(df_cam))

            df_all = pd.concat(df_all, axis=1)
            df_all.columns = ['_'.join(a) for a in df_all.columns.to_flat_index()]
            df_all.reset_index()

            df_all.to_csv(args.board_points_file)
                                
        camgroup.dump(calib_file)

        if args.debug:
            import matplotlib.pyplot as plt

            # generate images showing the detected points
            for vid, rows1 in zip(args.calibration_videos, rows):
                vid = os.path.join(args.base_path, vid)

                pn, fn = os.path.split(vid)
                fn, _ = os.path.splitext(fn)

                ngoodframes = len(rows1)

                with VideoCapture(vid) as cap:
                    for i in np.linspace(0, ngoodframes, num=args.ndebugimages, endpoint=False).astype(int):
                        fr = rows1[i]['framenum'][1]

                        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                        ret, frame = cap.read()

                        fig, ax = plt.subplots()
                        ax.imshow(frame)
                        ax.plot(rows1[i]['corners'][:,0,0], rows1[i]['corners'][:,0,1], 'ro')

                        outname1 = os.path.join(pn, '{0}-debug-{1:03d}.png'.format(fn, fr))
                        plt.savefig(outname1)
                        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Calibrate based on images of Charuco boards')

    parser.add_argument('config', nargs="+")

    parser.add_argument('--base_path', default='.',
                        help='Base path for data files')

    parser.add_argument('-nx', '--nsquaresx', type=int, 
                        help='Number of grid squares horizontally',
                        default=6)
    parser.add_argument('-ny', '--nsquaresy', type=int, 
                        help='Number of grid squares vertically',
                        default=6)
    parser.add_argument('-sz', '--square_length', type=float,
                        help = 'Size of square in mm',
                        default = 24.33)
    parser.add_argument('-mlen', '--marker_length', type=float,
                        help='Size of the Aruco marker in mm',
                        default=17)
    parser.add_argument('-mbits', '--marker_bits', type=int,
                        help='Information bits in the markers',
                        default=5)
    parser.add_argument('-dict','--dict_size', type=int,
                        help='Number of markers in the dictionary',
                        default=50)

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help='Display increasingly verbose output')

    parser.add_argument('--calibration_videos', action='extend', nargs='+',
                        help="Video files containing synchronized images of the Charuco board")

    parser.add_argument('--camera_names', action='extend', nargs='+',
                        help="Names for eaech of the cameras, in the same order as the videos")

    parser.add_argument('--force_calibration', default=False, action="store_true",
                        help="Run the calibration even if the calibration TOML file is present")

    parser.add_argument('--debug', type=bool, default=False,
                        help="Save debug images for the calibration")
    parser.add_argument('--ndebugimages', type=int, default=10,
                        help="Save debug images for the calibration")

    parser.add_argument('--showreprojection', type=bool, default=False,
                        help="Show debug images to test reprojection error")

    parser.add_argument('--calibration_file', help='TOML File to store the calibration')

    parser.add_argument('-s', '--sleap_files', action='extend', nargs='+',
                        help="SLP files containing points detected using Sleap.ai")
    parser.add_argument('--video_table', default=None,
                        help="CSV file containing a table of matching video files")
    
    parser.add_argument('-o', '--output_file', 
                        help="Name of the output CSV file with the triangulated points")

    args = parser.parse_args()

    if args.config is not None:
        for config1 in args.config:
            with open(config1, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
        
        args = parser.parse_args()

    do_calibration(args)

if __name__ == "__main__":
    main()

