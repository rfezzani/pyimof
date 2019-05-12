"""Common tools to optical flow algorithms

"""

import numpy as np
from scipy import ndimage as ndi
import skimage
from skimage.transform import pyramid_reduce, resize


def central_diff(p):
    """Forward difference scheme

    """
    p_y, p_x = np.gradient(p)
    p_x[:, 0] = 0
    p_x[:, -1] = 0
    p_y[0, :] = 0
    p_y[-1, :] = 0

    return p_x, p_y


def forward_diff(p):
    """Forward difference scheme

    """
    p_x = p.copy()
    p_x[:, 1:] -= p[:, :-1]
    p_x[:, 0] = 0
    p_x[:, -1] = 0

    p_y = p.copy()
    p_y[1:, :] -= p[:-1, :]
    p_y[0, :] = 0
    p_y[-1, :] = 0

    return p_x, p_y


def div(p1, p2):
    """Divergence of P=(p1, p2) using backward differece scheme.

    # """
    p1_x = p1.copy()
    p1_x[:, :-1] -= p1[:, 1:]
    # p1_x[:, -1] = 0

    p2_y = p2.copy()
    p2_y[:-1, :] -= p2[1:, :]
    # p2_y[-1, :] = 0

    return p1_x + p2_y


def warp(I, u, v, x=None, y=None, prefilter=True, mode='nearest'):
    """Image warping using the motion field (u, v)

    """
    if (x is None) or (y is None):
        nl, nc = I.shape
        y, x = np.meshgrid(np.arange(nl), np.arange(nc), indexing='ij')

    if prefilter:
        u_ = ndi.filters.median_filter(u, 3)
        v_ = ndi.filters.median_filter(v, 3)
    else:
        u_ = u
        v_ = v

    return ndi.map_coordinates(I, [y+v_, x+u_], order=2, mode=mode)


def upscale_flow(u, v, shape):
    """Rescale the values of the vector field (u, v) to the desired shape

    """

    nl, nc = u.shape
    sy, sx = shape[0]/nl, shape[1]/nc

    u = resize(u, shape, order=0, preserve_range=True,
               anti_aliasing=False)
    v = resize(v, shape, order=0, preserve_range=True,
               anti_aliasing=False)

    return sx*u, sy*v


def get_pyramid(I0, I1, downscale=2.0, min_size=16):
    """Image pyramid construction

    """

    if I0.shape != I1.shape:
        raise ValueError("Images should have the same size")

    pyramid = [(I0, I1)]
    size = min(I0.shape[:2])

    while size > min_size:
        J0 = pyramid_reduce(pyramid[-1][0], downscale, multichannel=False)
        J1 = pyramid_reduce(pyramid[-1][1], downscale, multichannel=False)
        pyramid.append((J0, J1))
        size = min(J0.shape[:2])

    return pyramid[::-1]


def coarse_to_fine(I0, I1, solver, downscale=2):
    """Generic coarse to fine solver

    """

    if (I0.ndim != 2) or (I1.ndim != 2):
        raise ValueError("Images should be grayscale.")

    pyramid = get_pyramid(skimage.img_as_float32(I0),
                          skimage.img_as_float32(I1), downscale)

    u = np.zeros_like(pyramid[0][0])
    v = np.zeros_like(u)

    u, v = solver(pyramid[0][0], pyramid[0][1], u, v)

    for J0, J1 in pyramid[1:]:
        u, v = upscale_flow(u, v, J0.shape)
        u, v = solver(J0, J1, u, v)

    return u, v
