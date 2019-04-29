"""Common tools to optical flow algorithms

"""

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import pyramid_reduce, resize


def warp(I, u, v, x=None, y=None, mode='nearest'):
    """Image warping using the motion field (u, v)

    """
    if (x is None) or (y is None):
        nl, nc = I.shape
        y, x = np.meshgrid(np.arange(nl), np.arange(nc), indexing='ij')

    return ndi.map_coordinates(I, [y+v, x+u], order=2, mode=mode)


def upscale_flow(u, v, shape):
    """Rescale the values of the vector field (u, v) to the desired shape

    """

    nl, nc = u.shape
    ax, ay = shape[0]/nl, shape[1]/nc

    u = resize(u, shape, preserve_range=True, anti_aliasing=False)
    v = resize(v, shape, preserve_range=True, anti_aliasing=False)

    return ax*u, ay*v


def get_pyramid(I0, I1, downscale=2, min_size=16):
    """Image pyramid construction

    """

    if I0.shape != I1.shape:
        raise ValueError("Images should have the same size")

    pyramid = [(I0, I1)]
    size = min(I0.shape[:2])

    while size > min_size:
        J0 = pyramid_reduce(pyramid[-1][0], downscale)
        J1 = pyramid_reduce(pyramid[-1][1], downscale)
        pyramid.append((J0, J1))
        size = min(J0.shape[:2])

    return pyramid[::-1]


def coarse_to_fine(I0, I1, solver, downscale=2):
    """Generic coarse to fine solver

    """

    if (I0.ndim != 2) or (I1.ndim != 2):
        raise ValueError("Images should be grayscale.")

    pyramid = get_pyramid(I0, I1, downscale)

    u = np.zeros_like(pyramid[0][0])
    v = np.zeros_like(u)

    u, v = solver(pyramid[0][0], pyramid[0][1], u, v)

    for J0, J1 in pyramid[1:]:
        u, v = upscale_flow(u, v, J0.shape)
        u, v = solver(J0, J1, u, v)

    return u, v
