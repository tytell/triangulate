import numpy as np
import pandas as pd
import aniposelib

def add_reprojected_points(pts, camgroup):

    # TODO: pts should have multiindex columns with camera at the first level
    # and x and y at the second

    # get the 3D coordinates
    ptsmatrix = []
    for cam1 in camgroup.get_names():
        pts1 = pts.loc[:, (cam1, ('x', 'y'))].to_numpy()
        ptsmatrix.append(pts1[np.newaxis, :,:])
    ptsmatrix = np.concatenate(ptsmatrix, axis=0)
    pts3d = camgroup.triangulate(ptsmatrix, progress=True, undistort=True)

    # and add them into the data frame
    col_idx = pd.MultiIndex.from_product([['3D'], ['x', 'y', 'z']],
                                    names=['camera','var'])
    dfpts3d = pd.DataFrame(pts3d, index=pts.index, 
                                columns=col_idx)
    pts = pd.concat((pts, dfpts3d), axis=1)

    # now get the reprojection error
    reproj = camgroup.project(pts3d)
    reproj = reproj.transpose((1,0,2))
    reproj = reproj.reshape((-1, 2*len(camgroup.get_names())))

    # and add those into the data frame
    col_idx = pd.MultiIndex.from_product([camgroup.get_names(), ['xr', 'yr']],
                                        names=['camera','var'])

    reproj = pd.DataFrame(reproj, 
                            index=pts.index, 
                            columns=col_idx)

    pts = pd.concat((pts, reproj), axis=1)

    return pts


