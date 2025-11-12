import os
import aniposelib
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

    logger.debug('Starting do_calibration')

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
            
            if pts1['corners'].ndim == 3 and pts1['corners'].shape[1] == 1:
                c1 = np.squeeze(pts1['corners'], axis=1)
                # NB - if we just run squeeze by itself, and pts1['coners']
                # has shape (1, 1, 2), we get c1 with shape (2,), which
                # will cause an error
            else:
                c1 = pts1['corners']

            logger.debug(f'{fr1=}, corners shape: {c1.shape}') 

            corner_df = pd.DataFrame(data = c1, 
                                    index=row_idx, 
                                    columns=col_idx)

            df_cam.append(corner_df)

        pts.append(pd.concat(df_cam))

    pts = pd.concat(pts, axis=1)
    
    return pts, camgroup

def get_triangulation_file_name(cfg, filename, withdate=True):
    parts = []
    parts.append(cfg['project_name'])
    if cfg['user'] is not None and \
            len(cfg['user']) > 0:
        parts.append(cfg['user'])
    
    if withdate:
        now = datetime.now().strftime('%Y-%m-%dT%H%M')

        parts.append(now)
    
    parts.append(filename)

    file = '-'.join(parts)

    return file

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


def refine_calibration(camgroup, pts, max_err, nodes='all', outfile=None):
    npt = pts.shape[0]
    good = ~pts.loc[:,(slice(None), ['x', 'y'])].isna().any(axis=1)
    pts = pts.loc[good, :]
    print(f'{pts.shape[0]}/{npt} points visible in all cameras')

    if nodes != 'all':
        pts = pts.loc[(slice(None), slice(None), nodes), :]
        print('{}/{} ({:2.0f}%) points in selected nodes'.format(pts.shape[0], npt, np.round(pts.shape[0]/npt*100)))

    good = pts.loc[:, (slice(None), 'err')].max(axis=1, skipna=False) < max_err

    pts = pts.loc[good, (camgroup.get_names(), ['x', 'y'])]
    print('{}/{} ({:2.0f}%) points below maximum error'.format(pts.shape[0], npt, np.round(pts.shape[0]/npt*100)))

    calibpts = pts.to_numpy(dtype=float)
    calibpts = np.reshape(calibpts, [-1, 3, 2]).transpose((1, 0, 2))

    camgroup1 = deepcopy(camgroup)
    camgroup1.bundle_adjust_iter(calibpts, extra=None,
                            n_iters=6, start_mu=15, end_mu=1,
                            max_nfev=200, ftol=1e-4,
                            n_samp_iter=200, n_samp_full=1000,
                            error_threshold=0.3, only_extrinsics=False,
                            verbose=True)    

    if outfile is not None:
        camgroup1.dump(outfile)

    return camgroup1
