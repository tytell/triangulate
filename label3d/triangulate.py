import os
import aniposelib
import argparse
import yaml
import cv2

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

import pandas as pd
import re
import fnmatch

from copy import copy, deepcopy
from warnings import warn

def reorganize_sleap_csv(data):

    # we assume the columns have the structure
    #   - track (irrelevant for these purposes)
    #   - frame_idx (frame number)
    #   - instance.score
    #   - point.x, point.y, point.score,
    #       where "point" is the name of a point and it repeats for
    #       as many points are labeled
    assert data.columns[0] == 'track', "Columns are not organized the way we expect"
    assert data.columns[1] == 'frame_idx', "Columns are not organized the way we expect"
    assert data.columns[2] == 'instance.score', "Columns are not organized the way we expect"

    assert re.match("\w+\.x", data.columns[3]) is not None, "Columns are not organized the way we expect"

    col_idx = pd.MultiIndex.from_tuples(data.columns[3:].str.split('.').to_list(),
        names = ('point', 'var'))    

    data = data.set_index('frame_idx')

    point_data = data.drop(['track', 'instance.score'], axis=1)
    point_data.columns = col_idx
    point_data = point_data.stack(level = "point", future_stack=True)

    inst_score = data['instance.score']

    point_data = point_data.join(inst_score, on='frame_idx')

    return point_data
    
def triangulate_points(camgroup, pts):
    ncam = len(camgroup.get_names())

    # get rid of any columns that might have come from a previous reprojection
    pts = pts.loc[:, (camgroup.get_names(), ['x','y'])]

    # triangulate to 3D
    ptsmatrix = pts.to_numpy().reshape((-1, ncam, 2))
    ptsmatrix = ptsmatrix.transpose((1,0,2)).astype(np.float64)

    pts3d = camgroup.triangulate(ptsmatrix, progress=True, undistort=True)

    col_idx = pd.MultiIndex.from_product([['3D'], ['x', 'y', 'z']],
                                         names=['camera','var'])
    dfpts3d = pd.DataFrame(pts3d, index=pts.index, 
                              columns=col_idx)
    pts = pd.concat((pts, dfpts3d), axis=1)

    # now compute the reprojection error
    reproj = camgroup.project(pts3d)
    reproj = reproj.transpose((1,0,2))
    reproj = reproj.reshape((-1, ncam*2))

    col_idx = pd.MultiIndex.from_product([camgroup.get_names(), ['Xr', 'Yr']],
                                         names=['camera','var'])

    reproj = pd.DataFrame(reproj, 
                          index=pts.index, 
                          columns=col_idx)

    pts = pd.concat((pts, reproj), axis=1)

    errs = []
    for cam in camgroup.get_names():
        xy = pts.loc[:,(cam,['x','y'])].to_numpy(dtype=float)
        XYr = pts.loc[:,(cam,['Xr','Yr'])].to_numpy(dtype=float)

        errs.append(
            pd.DataFrame(np.linalg.norm(XYr - xy, axis=1), index=pts.index,
                    columns=pd.MultiIndex.from_product([[cam], ['err']],
                                                names=['camera','var']))
        )

    errs = pd.concat(errs, axis=1)
    pts = pd.concat((pts, errs), axis=1)

    return pts


def plot_reprojected_points(pts, i, videonames=None, videopath=None, zoom=True):
    fr = pts.iloc[i:i+1].index.get_level_values('frame')[0]
    try:
        vid = pts.iloc[i:i+1].index.get_level_values('video')[0]
        q = f'frame == {fr} and video == "{vid}"'
    except KeyError:
        q = f'frame == {fr}'

    print(q)
    pts1 = pts.query(q)

    camnames = list(pts1.columns.get_level_values('camera').unique())
    camnames = [cn1 for cn1 in camnames if cn1 != '3D']

    if videonames is None:
        videoname = list(pts.iloc[i:i+1].index.get_level_values('video'))[0]
        videonames = []
        for cam1 in camnames:
            fn1, _ = re.subn('CAMERA', cam1, videoname)
            videonames.append(fn1)
    
    if videopath is not None:
        videonames = [[os.path.join(videopath, vn1)] for vn1 in videonames]

    try:
        cap = [cv2.VideoCapture(vid1[0]) for vid1 in videonames]

        fig, ax = plt.subplots(ncols=len(cap), nrows=1) #, sharex=True, sharey=True)

        for cam1, cap1, ax1 in zip(camnames, cap, ax):
            cap1.set(1, fr)
            ret, frame1 = cap1.read()

            ax1.imshow(frame1)

            x1 = pts1[(cam1, 'x')].array
            y1 = pts1[(cam1, 'y')].array
            xr1 = pts1[(cam1, 'Xr')].array
            yr1 = pts1[(cam1, 'Yr')].array

            ax1.plot(x1,y1, 'ro')
            ax1.plot(xr1,yr1, 'y+')
            ax1.plot(np.vstack((x1, xr1)), 
                     np.vstack((y1, yr1)), 'y-')

            xx = pts1.loc[(slice(None)), (cam1, ['x','Xr'])].stack()
            yy = pts1.loc[(slice(None)), (cam1, ['y','Yr'])].stack()

            if zoom:
                ax1.set_xlim(pd.concat([xx.min(), xx.max()]).to_numpy() + np.array([-50, 50]))
                ax1.set_ylim(pd.concat([yy.min(), yy.max()]).to_numpy() + np.array([-50, 50]))        
            ax1.invert_yaxis()
            ax1.axis('off')
    finally:
        for c1 in cap:
            c1.release()
    
    return fig

def separate_video_and_camera(vidname, camnames):
    fn1 = re.sub(r'\\', '/', vidname)
    fn1 = os.path.basename(fn1)

    for cam1 in camnames:
        fn1, nsub = re.subn(cam1, 'CAMERA', fn1)
        if nsub == 1:
            matched_camera = cam1
            break
    else:
        matched_camera = None

    return fn1, matched_camera

