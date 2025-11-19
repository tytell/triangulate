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

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_basis(axesdata, order_axes='length', make_right_handed=True,
              origin_name='origin', point_names=['X', 'Y', 'Z']):
    
    origin = axesdata.loc[(slice(None), origin_name), :]
    
    axesdata0 = axesdata.loc[(slice(None), point_names), :]
    axesdata0 = axesdata0  - origin.values

    axesdata0.index = pd.Index(['x', 'y', 'z'])
    axesdata0.columns = pd.Index(['xorig', 'yorig', 'zorig'])

    if order_axes == 'length':
        axlen = np.sqrt(np.square(axesdata0.T).sum())
        axlen = axlen.sort_values(ascending=False)

        axesdata0 = axesdata0.reindex(axlen.index)

        axlenstr = ['{}={:.1f}'.format(n,v) for n,v in zip(axlen.index, axlen)]
        print('Ordering axes by length: {}'.format(', '.join(axlenstr)))
    elif order_axes is None:
        pass

    h = np.cross(axesdata0.iloc[0], axesdata0.iloc[1]) @ axesdata0.iloc[2]
    if make_right_handed and (h < 0):
        print('Axes matrix is left handed! Flipping {} axis'.format(axesdata0.index[2]))

        axesdata0.iloc[2] = -axesdata0.iloc[2]

    R = gram_schmidt(axesdata0.T.to_numpy())

    R = pd.DataFrame(R, columns = axesdata0.index)

    # this reorders the columns in alphabetical order
    R = R.reindex(columns=R.columns.sort_values())

    R.index = pd.Index(['x', 'y', 'z'])
    origindf = origin.copy()
    origindf.index = origindf.index.droplevel(0)
    origindf.columns = R.columns

    return pd.concat((origindf, R))

def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Get the number of vectors.

    assert(A.ndim == 2)

    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A

def rename_columns(df, columns, inplace=False):
    """Rename dataframe columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe.
    columns : dict-like
        Alternative to specifying axis. If `df.columns` is
        :obj: `pandas.MultiIndex`-object and has a few levels, pass equal-size tuples.

    Returns
    -------
    pandas.DataFrame or None
        Returns dataframe with modifed columns or ``None`` (depends on `inplace` parameter value).
    
    Examples
    --------
    >>> columns = pd.Index([1, 2, 3])
    >>> df = pd.DataFrame([[1, 2, 3], [10, 20, 30]], columns=columns)
    ...     1   2   3
    ... 0   1   2   3
    ... 1  10  20  30
    >>> rename_columns(df, columns={1 : 10})
    ...    10   2   3
    ... 0   1   2   3
    ... 1  10  20  30
    
    MultiIndex
    
    >>> columns = pd.MultiIndex.from_tuples([("A0", "B0", "C0"), ("A1", "B1", "C1"), ("A2", "B2", "")])
    >>> df = pd.DataFrame([[1, 2, 3], [10, 20, 30]], columns=columns)
    >>> df
    ...    A0  A1  A2
    ...    B0  B1  B2
    ...    C0  C1
    ... 0   1   2   3
    ... 1  10  20  30
    >>> rename_columns(df, columns={("A2", "B2", "") : ("A3", "B3", "")})
    ...    A0  A1  A3
    ...    B0  B1  B3
    ...    C0  C1
    ... 0   1   2   3
    ... 1  10  20  30
    """
    columns_new = []
    for col in df.columns.values:
        if col in columns:
            columns_new.append(columns[col])
        else:
            columns_new.append(col)
    columns_new = pd.Index(columns_new, tupleize_cols=True)

    if inplace:
        df.columns = columns_new
    else:
        df_new = df.copy()
        df_new.columns = columns_new
        return df_new

def rotate_axes(data, basis):
    data3d = data.loc[:, ('3D', ['x', 'y', 'z'])]

    data_rot = rename_columns(data, {('3D', 'x'): ('3D', 'x0'),
                                       ('3D', 'y'): ('3D', 'y0'),
                                       ('3D', 'z'): ('3D', 'z0')})
    
    origin = basis.loc['origin', :]
    R = basis.loc[['x', 'y', 'z'], :]
    
    data3d_r = (data3d.to_numpy() - origin.values) @ R

    col_idx = pd.MultiIndex.from_product([['3D'], ['x','y','z']])
    data3d_r = pd.DataFrame(data3d_r.values, index=data_rot.index, columns=col_idx)

    return pd.concat((data_rot, data3d_r), axis=1)
