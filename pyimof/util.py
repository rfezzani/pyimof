# coding: utf-8
"""Common tools to optical flow algorithms.

"""

import numpy as np
import skimage
from skimage.transform import pyramid_reduce, resize


def central_diff(p):
    """Central difference scheme.

    Parameters
    ----------
    p : 2D ndarray
        The array to be processed.

    Returns
    -------
    p_x : 2D ~numpy.ndarray
        The horizontal gradient component.
    p_y : 2D ~numpy.ndarray
        The vertical gradient component.

    """
    p_y, p_x = np.gradient(p)
    p_x[:, 0] = 0
    p_x[:, -1] = 0
    p_y[0, :] = 0
    p_y[-1, :] = 0

    return p_x, p_y


def forward_diff(p):
    """Forward difference scheme

    Parameters
    ----------
    p : 2D ~numpy.ndarray
        The array to be processed.

    Returns
    -------
    p_x : 2D ~numpy.ndarray
        The horizontal gradient component.
    p_y : 2D ~numpy.ndarray
        The vertical gradient component.

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
    """Divergence of P=(p1, p2) using backward differece scheme:

    div(P) = p1_x + p2_y

    Parameters
    ----------
    p1 : 2D ~numpy.ndarray
        The first component to be processed.
    p2 : 2D ~numpy.ndarray
        The second component to be processed.

    Returns
    -------
    div_p : 2D ~numpy.ndarray
        The divergence of P=(p1, p2).

    """
    p1_x = p1.copy()
    p1_x[:, :-1] -= p1[:, 1:]

    p2_y = p2.copy()
    p2_y[:-1, :] -= p2[1:, :]

    div_p = p1_x + p2_y

    return div_p


def resize_flow(u, v, shape):
    """Rescale the values of the vector field (u, v) to the desired shape.

    The values of the output vector field are scaled to the new
    resolution.

    Parameters
    ----------
    u : 2D ~numpy.ndarray
        The horizontal component of the motion field.
    v : 2D ~numpy.ndarray
        The vertical component of the motion field.
    shape : Iterable
        Couple of integers representing the output shape.

    Returns
    -------
    ru : 2D ~numpy.ndarray
        The resized and rescaled horizontal component of the motion
        field.
    rv : 2D ~numpy.ndarray
        The resized and rescaled vertical component of the motion
        field.

    """

    nl, nc = u.shape
    sy, sx = shape[0]/nl, shape[1]/nc

    u = resize(u, shape, order=0, preserve_range=True,
               anti_aliasing=False)
    v = resize(v, shape, order=0, preserve_range=True,
               anti_aliasing=False)

    ru, rv = sx*u, sy*v

    return ru, rv


def get_pyramid(I, downscale=2.0, nlevel=10, min_size=16):
    """Construct image pyramid.

    Parameters
    ----------
    I : 2D or 3D ~numpy.ndarray
        The image to be preprocessed.
    downscale : float
        The pyramid downscale factor (default: 2)
    nlevel : int
        The maximum number of pyramid levels (default: 10).
    min_size : int
        The minimum size for any dimension of the pyramid levels
        (default: 16).

    Returns
    -------
    pyramid : list
        The coarse to fine images pyramid.

    """

    pyramid = [I]
    size = min(I.shape)
    count = 1

    while (count < nlevel) and (size > min_size):
        J = pyramid_reduce(pyramid[-1], downscale, multichannel=False)
        pyramid.append(J)
        size = min(J.shape)
        count += 1

    return pyramid[::-1]


def coarse_to_fine(I0, I1, solver, downscale=2, nlevel=10, min_size=16):
    """Generic coarse to fine solver.

    Parameters
    ----------
    I0 : 2D ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : 2D ~numpy.ndarray
        The second gray scale image of the sequence.
    solver : callable
        The solver applyed at each pyramid level.
    downscale : float
        The pyramid downscale factor (default: 2)
    nlevel : int
        The maximum number of pyramid levels (default: 10).
    min_size : int
        The minimum size for any dimension of the pyramid levels
        (default: 16).

    Returns
    -------
    u : 2D ~numpy.ndarray
        the horizontal component of the estimated optical flow.
    v : 2D ~numpy.ndarray
        the vertical component of the estimated optical flow.

    """

    if (I0.ndim != 2) or (I1.ndim != 2):
        raise ValueError("Images should be grayscale.")

    if I0.shape != I1.shape:
        raise ValueError("Images should have the same shape")

    pyramid = list(zip(get_pyramid(skimage.img_as_float32(I0),
                                   downscale, nlevel, min_size),
                       get_pyramid(skimage.img_as_float32(I1),
                                   downscale, nlevel, min_size)))

    u = np.zeros_like(pyramid[0][0])
    v = np.zeros_like(u)

    u, v = solver(pyramid[0][0], pyramid[0][1], u, v)

    for J0, J1 in pyramid[1:]:
        u, v = resize_flow(u, v, J0.shape)
        u, v = solver(J0, J1, u, v)

    return u, v
