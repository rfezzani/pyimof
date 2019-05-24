# coding: utf-8
"""Functions to read and write '.flo' Middleburry file format.

"""

import os
import warnings
import numpy as np

TAG_FLOAT = 202021.25  # check for this when READING the file
TAG_STRING = 'PIEH'


def flowrite(u, v, fname):
    """Write a given flow to the Middleburry file format .flo

    Parameters
    ----------
    u : ~numpy.ndarray
        The horizontal component of the estimated optical flow.
    v : ~numpy.ndarray
        The vertical component of the estimated optical flow.
    fname: str
        The target file name. The '.flo' extension is appended if
        necessary.

    """

    fname = os.path.abspath(fname)
    if not fname.endswith('.flo'):
        warnings.warn("Saving to {}".format(fname+'.flo'))
        fname += '.flo'

    h, w = u.shape

    img = np.zeros((h, w, 2))
    img[:, :, 0] = v
    img[:, :, 1] = u

    with open(fname, 'wb') as f:

        # write the header
        np.array(TAG_STRING).tofile(f)
        np.array([w, h], dtype=np.int32).tofile(f)

        # arrange into matrix form
        tmp = np.zeros((h, w*2))

        tmp[:, ::2] = v
        tmp[:, 1::2] = u
        tmp.astype('float32').tofile(f)


def floread(fname):
    """Read a Middleburry .flo file.

    Parameters
    ----------
    fname: str
        The file name.

    Returns
    -------
    u, v : tuple[~numpy.ndarray]
        The horizontal and vertical components of the estimated
        optical flow.

    """

    fname = os.path.abspath(fname)
    if not fname.endswith('.flo'):
        warnings.warn("Reading flow from {}".format(fname+'.flo'))
        fname += '.flo'

    with open(fname, 'rb') as f:

        tag = np.fromfile(f, dtype=np.float32, count=1)[0]

        # sanity check
        if not tag == TAG_FLOAT:
            raise ValueError('{}: wrong tag (big-endian issue?)'.format(fname))

        w, h = np.fromfile(f, dtype=np.int32, count=2)

        if (w < 1) or (h < 1):
            raise ValueError('{}: wrong shape ({}, {})'.format(fname, w, h))

        buf = np.fromfile(f, dtype=np.float32).reshape((h, w*2))
        v = buf[:, ::2]
        u = buf[:, 1::2]

    return u, v
