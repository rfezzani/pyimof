# coding: utf-8
"""Common tools to optical flow algorithms.

"""

import numpy as np
import skimage
from skimage.transform import pyramid_reduce, resize


def tv_regularize(x, tau=0.3, dt=0.2, max_iter=100, p=None, g=None):
    """Toltal variation regularization using Chambolle algorithm [1]_.

    Parameters
    ----------
    x : ~numpy.ndarray
        The target array.
    tau : float
        Tightness parameter. It should have a small value in order to
        maintain attachement and regularization parts in
        correspondence.
    dt : float
        Time step of the numerical scheme. Convergence is proved for
        values dt < 0.125, but it can be larger for faster
        convergence.
    max_iter : int
        Maximum number of iteration.
    p : ~numpy.ndarray
        Optional buffer array of shape (x.ndim, ) + x.shape.
    g : ~numpy.ndarray
        Optional buffer array of shape (x.ndim, ) + x.shape.

    References
    ----------
    .. [1] A. Chambolle, An algorithm for total variation minimization and
           applications, Journal of Mathematical Imaging and Vision,
           Springer, 2004, 20, 89-97.

    """
    f = dt / tau
    out = x

    s_g = [slice(None), ] * g.ndim
    s_p = [slice(None), ] * p.ndim
    s_d = [slice(None), ] * (p.ndim-1)

    for _ in range(max_iter):
        for ax in range(x.ndim):
            s_g[0] = ax
            s_g[ax+1] = slice(0, -1)
            g[tuple(s_g)] = np.diff(out, axis=ax)
            s_g[ax+1] = slice(None)

        norm = np.sqrt((g ** 2).sum(0))[np.newaxis, ...]
        norm *= f
        norm += 1.
        p -= dt * g
        p /= norm

        # d will be the (negative) divergence of p
        d = -p.sum(0)
        for ax in range(x.ndim):
            s_d[ax] = slice(1, None)
            s_p[ax+1] = slice(0, -1)
            s_p[0] = ax
            d[tuple(s_d)] += p[tuple(s_p)]
            s_d[ax] = slice(None)
            s_p[ax+1] = slice(None)

        out = x + d
    return out


def flow_tv_regularize(x, tau=0.3, dt=0.2, max_iter=100, p=None, g=None):
    """Toltal variation regularization using Chambolle algorithm [1]_.

    Parameters
    ----------
    x : ~numpy.ndarray
        The target array.
    tau : float
        Tightness parameter. It should have a small value in order to
        maintain attachement and regularization parts in
        correspondence.
    dt : float
        Time step of the numerical scheme. Convergence is proved for
        values dt < 0.125, but it can be larger for faster
        convergence.
    max_iter : int
        Maximum number of iteration.
    p : ~numpy.ndarray
        Optional buffer array of shape (x.ndim, x.ndim, ) + x.shape.
    g : ~numpy.ndarray
        Optional buffer array of shape (x.ndim, ) + x.shape.

    References
    ----------
    .. [1] A. Chambolle, An algorithm for total variation minimization and
           applications, Journal of Mathematical Imaging and Vision,
           Springer, 2004, 20, 89-97.

    """
    f = dt / tau
    out = x.copy()

    s_g = [slice(None), ] * g.ndim
    s_p = [slice(None), ] * p.ndim
    s_d = [slice(None), ] * (p.ndim-2)

    for idx in range(x.shape[0]):
        s_p[0] = idx
        for _ in range(max_iter):
            for ax in range(x.shape[0]):
                s_g[0] = ax
                s_g[ax+1] = slice(0, -1)
                g[tuple(s_g)] = np.diff(out[idx], axis=ax)
                s_g[ax+1] = slice(None)

            norm = np.sqrt((g ** 2).sum(0))[np.newaxis, ...]
            norm *= f
            norm += 1.
            p[idx] -= dt * g
            p[idx] /= norm

            # d will be the (negative) divergence of p[idx]
            d = -p[idx].sum(0)
            for ax in range(x.shape[0]):
                s_p[1] = ax
                s_p[ax+2] = slice(0, -1)
                s_d[ax] = slice(1, None)
                d[tuple(s_d)] += p[tuple(s_p)]
                s_p[ax+2] = slice(None)
                s_d[ax] = slice(None)

            out[idx] = x[idx] + d
    return out


def resize_flow(u, v, shape):
    """Rescale the values of the vector field (u, v) to the desired shape.

    The values of the output vector field are scaled to the new
    resolution.

    Parameters
    ----------
    u : ~numpy.ndarray
        The horizontal component of the motion field.
    v : ~numpy.ndarray
        The vertical component of the motion field.
    shape : iterable
        Couple of integers representing the output shape.

    Returns
    -------
    ru, rv : tuple[~numpy.ndarray]
        The resized and rescaled horizontal and vertical components of
        the motion field.

    """

    nl, nc = u.shape
    sy, sx = shape[0]/nl, shape[1]/nc

    u = resize(u, shape, order=0, preserve_range=True,
               anti_aliasing=False)
    v = resize(v, shape, order=0, preserve_range=True,
               anti_aliasing=False)

    ru, rv = sx*u, sy*v

    return ru, rv


def nd_resize_flow(flow, shape):
    """Rescale the values of the vector field (u, v) to the desired shape.

    The values of the output vector field are scaled to the new
    resolution.

    Parameters
    ----------
    u : ~numpy.ndarray
        The horizontal component of the motion field.
    v : ~numpy.ndarray
        The vertical component of the motion field.
    shape : iterable
        Couple of integers representing the output shape.

    Returns
    -------
    ru, rv : tuple[~numpy.ndarray]
        The resized and rescaled horizontal and vertical components of
        the motion field.

    """

    scale = np.array([n/o for n, o in zip(shape, flow.shape[1:])])

    for _ in shape:
        scale = scale[..., np.newaxis]

    return scale*resize(flow, (flow.shape[0],)+shape, order=0,
                        preserve_range=True, anti_aliasing=False)


def get_pyramid(I, downscale=2.0, nlevel=10, min_size=16):
    """Construct image pyramid.

    Parameters
    ----------
    I : ~numpy.ndarray
        The image to be preprocessed (Gray scale or RGB).
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.

    Returns
    -------
    pyramid : list[~numpy.ndarray]
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
    I0 : ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : ~numpy.ndarray
        The second gray scale image of the sequence.
    solver : callable
        The solver applyed at each pyramid level.
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.

    Returns
    -------
    u, v : tuple[~numpy.ndarray]
        The horizontal and vertical components of the estimated
        optical flow.

    """

    if (I0.ndim != 2) or (I1.ndim != 2):
        raise ValueError("Only grayscale images are supported.")

    if I0.shape != I1.shape:
        raise ValueError("Input images should have the same shape")

    pyramid = list(zip(get_pyramid(skimage.img_as_float32(I0),
                                   downscale, nlevel, min_size),
                       get_pyramid(skimage.img_as_float32(I1),
                                   downscale, nlevel, min_size)))

    flow = np.zeros((pyramid[0][0].ndim, ) + pyramid[0][0].shape)

    flow = solver(pyramid[0][0], pyramid[0][1], flow)

    for J0, J1 in pyramid[1:]:
        flow = nd_resize_flow(flow, J0.shape)
        flow = solver(J0, J1, flow)

    return flow
