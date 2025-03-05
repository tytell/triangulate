import os
import argparse
import yaml
import aniposelib
import cv2
import numpy as np
import pandas as pd

from .reproject import add_reprojected_points

from contextlib import contextmanager
@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()

def do_calibration(args, verbose=True, debug=True, ndebugimages=10):
    ## Run the calibration if necessary
    calib_file = os.path.join(args['base_path'], args['calibration_file'])

    if verbose > 0:
        if not os.path.exists(calib_file):
            print(f"No calibration file found. Running calibration")
        else:
            print("force_calibration is True. Running calibration")

    board = aniposelib.boards.CharucoBoard(squaresX=args['nsquaresx'],
                                        squaresY=args['nsquaresy'],
                                        square_length=args['square_length'],
                                        marker_length=args['marker_length'],
                                        marker_bits=args['marker_bits'],
                                        dict_size=args['dict_size'])

    assert len(args['calibration_videos']) == len(args['camera_names']), \
        f'Number of calibration videos is different than number of camera names'
    camgroup = aniposelib.cameras.CameraGroup.from_names(args['camera_names'])

    vidnames = [[os.path.join(args['base_path'], vid)] for vid in args['calibration_videos']]
    err, rows = camgroup.calibrate_videos(vidnames, board, 
                            init_intrinsics=True, init_extrinsics=True, 
                            verbose=verbose > 0)

    # make a data frame for the detected points
    pts = []
    for cam1, rows_cam in zip(args['camera_names'], rows):
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
        pts.append(pd.concat(df_cam))

    pts = pd.concat(pts, axis=1)
    
    pts = add_reprojected_points(pts, camgroup)
                            
    camgroup.dump(calib_file)

    return pts


def save_detected_points(args, pts):
    # save the detected points
    pts = []
    for cam1, rows_cam in zip(args['camera_names'], rows):
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
        pts.append(pd.concat(df_cam))

    pts = pd.concat(pts, axis=1)
    pts.columns = ['_'.join(a) for a in pts.columns.to_flat_index()]
    pts.reset_index()

    pts.to_csv(args['board_points_file'])

def generate_debug_images(args, pts):

    import matplotlib.pyplot as plt

    # generate images showing the detected points
    for vid, rows1 in zip(args['calibration_videos'], rows):
        vid = os.path.join(args['base_path'], vid)

        pn, fn = os.path.split(vid)
        fn, _ = os.path.splitext(fn)

        ngoodframes = len(rows1)

        with VideoCapture(vid) as cap:
            for i in np.linspace(0, ngoodframes, num=args['ndebugimages'], endpoint=False).astype(int):
                fr = rows1[i]['framenum'][1]

                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ret, frame = cap.read()

                fig, ax = plt.subplots()
                ax.imshow(frame)
                ax.plot(rows1[i]['corners'][:,0,0], rows1[i]['corners'][:,0,1], 'ro')

                outname1 = os.path.join(pn, '{0}-debug-{1:03d}.png'.format(fn, fr))
                plt.savefig(outname1)
                plt.close(fig)
