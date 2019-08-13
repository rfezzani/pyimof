# coding: utf-8
"""Collection of optical flow algorithms.

"""

from functools import partial
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import warp

from .util import coarse_to_fine, flow_tv_regularize


def _tvl1(I0, I1, flow0, dt, lambda_, tau, nwarp, niter, tol, prefilter):
    """TV-L1 solver for optical flow estimation.

    Parameters
    ----------
    I0 : ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : ~numpy.ndarray
        The second gray scale image of the sequence.
    flow0 : ~numpy.ndarray
        Initial vector field.
    dt : float
        Time step of the numerical scheme. Convergence is proved for
        values dt < 0.125, but it can be larger for faster
        convergence.
    lambda_ : float
        Attachement parameter. The smaller this parameter is,
        the smoother is the solutions.
    tau : float
        Tightness parameter. It should have a small value in order to
        maintain attachement and regularization parts in
        correspondence.
    nwarp : int
        Number of times I1 is warped.
    niter : int
        Number of fixed point iteration.
    tol : float
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool
        whether to prefilter the estimated optical flow before each
        image warp.

    Returns
    -------
    u, v : tuple[~numpy.ndarray]
        The horizontal and vertical components of the estimated
        optical flow.

    """

    grid = np.array(
        np.meshgrid(*[np.arange(n) for n in I0.shape], indexing='ij'))

    f0 = lambda_*tau
    tol *= I0.size

    flow = flow0

    g = np.zeros((I0.ndim, ) + I0.shape)
    proj = np.zeros((I0.ndim, I0.ndim, ) + I0.shape)

    for _ in range(nwarp):
        if prefilter:
            flow = ndi.filters.median_filter(flow, [1]+I0.ndim*[3])

        wI1 = warp(I1, grid+flow, mode='nearest')
        grad = np.array(np.gradient(wI1))
        NI = (grad*grad).sum(0)
        NI[NI == 0] = 1

        rho_0 = wI1 - I0 - (grad*flow0).sum(0)

        for _ in range(niter):

            # Data term

            rho = rho_0 + (grad*flow).sum(0)

            idx = abs(rho) <= f0*NI

            flow_ = flow

            flow_[:, idx] -= rho[idx]*grad[:, idx]/NI[idx]

            idx = ~idx
            srho = f0*np.sign(rho[idx])

            flow_[:, idx] -= srho*grad[:, idx]

            # Regularization term

            flow = flow_tv_regularize(flow_, tau, dt, 2, proj, g)

        flow0 -= flow
        if (flow0*flow0).sum() < tol:
            break

        flow0 = flow

    return flow


def tvl1(I0, I1, dt=0.2, lambda_=15, tau=0.3, nwarp=5, niter=10,
         tol=1e-4, prefilter=False):
    """Coarse to fine TV-L1 optical flow estimator. A popular algorithm
    intrudced by Zack et al. [1]_, improved in [2]_ and detailed in [3]_.

    Parameters
    ----------
    I0 : ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : ~numpy.ndarray
        The second gray scale image of the sequence.
    dt : float
        Time step of the numerical scheme. Convergence is proved for
        values dt < 0.125, but it can be larger for faster
        convergence.
    lambda_ : float
        Attachement parameter. The smaller this parameter is,
        the smoother is the solutions.
    tau : float
        Tightness parameter. It should have a small value in order to
        maintain attachement and regularization parts in
        correspondence.
    nwarp : int
        Number of times I1 is warped.
    niter : int
        Number of fixed point iteration.
    tol : float
        Tolerance used as stopping criterion based on the L² distance
        between two consecutive values of (u, v).
    prefilter : bool
        whether to prefilter the estimated optical flow before each
        image warp.

    Returns
    -------
    u, v : tuple[~numpy.ndarray]
        The horizontal and vertical components of the estimated
        optical flow.

    References
    ----------
    .. [1] Zach, C., Pock, T., & Bischof, H. (2007, September). A
       duality based approach for realtime TV-L 1 optical flow. In Joint
       pattern recognition symposium (pp. 214-223). Springer, Berlin,
       Heidelberg.
    .. [2] Wedel, A., Pock, T., Zach, C., Bischof, H., & Cremers,
       D. (2009). An improved algorithm for TV-L 1 optical flow. In
       Statistical and geometrical approaches to visual motion analysis
       (pp. 23-45). Springer, Berlin, Heidelberg.
    .. [3] Pérez, J. S., Meinhardt-Llopis, E., & Facciolo,
       G. (2013). TV-L1 optical flow estimation. Image Processing On
       Line, 2013, 137-150.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> import pyimof
    >>> I0, I1 = pyimof.data.yosemite()
    >>> u, v = pyimof.solvers.tvl1(I0, I1)
    >>> pyimof.display.plot(u, v)
    >>> plt.show()

    """

    solver = partial(_tvl1, dt=dt, lambda_=lambda_, tau=tau,
                     nwarp=nwarp, niter=niter, tol=tol,
                     prefilter=prefilter)

    return coarse_to_fine(I0, I1, solver)


def _ilk(I0, I1, flow0, rad, nwarp, gaussian, prefilter):
    """Iterative Lucas-Kanade (iLK) solver for optical flow estimation.

    Parameters
    ----------
    I0 : ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : ~numpy.ndarray
        The second gray scale image of the sequence.
    u0 : ~numpy.ndarray
        Initialization for the horizontal component of the vector
        field.
    v0 : ~numpy.ndarray
        Initialization for the vertical component of the vector
        field.
    rad : int
        Radius of the window considered around each pixel.
    nwarp : int
        Number of times I1 is warped.
    gaussian : bool
        if True, gaussian kernel is used otherwise uniform kernel is
        used.
    prefilter : bool
        whether to prefilter the estimated optical flow before each
        image warp.

    Returns
    -------
    u, v : tuple[~numpy.ndarray]
        The horizontal and vertical components of the estimated
        optical flow.

    """

    nl, nc = I0.shape
    y, x = np.meshgrid(np.arange(nl), np.arange(nc), indexing='ij')

    size = 2*rad+1

    if gaussian:
        filter_func = partial(ndi.gaussian_filter, sigma=size/4, mode='mirror')
    else:
        filter_func = partial(ndi.uniform_filter, size=size, mode='mirror')

    v0, u0 = flow0
    u = u0.copy()
    v = v0.copy()

    for _ in range(nwarp):
        if prefilter:
            u = ndi.filters.median_filter(u, 3)
            v = ndi.filters.median_filter(v, 3)

        wI1 = warp(I1, np.array([y+v, x+u]), mode='nearest')
        Iy, Ix = np.gradient(wI1)
        It = wI1 - I0 - u*Ix - v*Iy

        J11 = Ix*Ix
        J12 = Ix*Iy
        J22 = Iy*Iy
        J13 = Ix*It
        J23 = Iy*It

        filter_func(J11, output=J11)
        filter_func(J12, output=J12)
        filter_func(J22, output=J22)
        filter_func(J13, output=J13)
        filter_func(J23, output=J23)

        detA = -(J11*J22 - J12*J12)
        idx = abs(detA) < 1e-14
        detA[idx] = 1

        u = (J13*J22 - J12*J23)/detA
        v = (J23*J11 - J12*J13)/detA

        u[idx] = 0
        v[idx] = 0

    return np.array([v, u])


def ilk(I0, I1, rad=7, nwarp=10, gaussian=True, prefilter=False):
    """Coarse to fine iterative Lucas-Kanade (iLK) optical flow
    estimator. A fast and robust algorithm developped by Le Besnerais
    and Champagnat [4]_ and improved in [5]_..

    Parameters
    ----------
    I0 : ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : ~numpy.ndarray
        The second gray scale image of the sequence.
    rad : int
        Radius of the window considered around each pixel.
    nwarp : int
        Number of times I1 is warped.
    gaussian : bool
        if True, gaussian kernel is used otherwise uniform kernel is
        used.
    prefilter : bool
        whether to prefilter the estimated optical flow before each
        image warp.

    Returns
    -------
    u, v : tuple[~numpy.ndarray]
        The horizontal and vertical components of the estimated
        optical flow.

    References
    ----------
    .. [4] Le Besnerais, G., & Champagnat, F. (2005, September). Dense
       optical flow by iterative local window registration. In IEEE
       International Conference on Image Processing 2005 (Vol. 1,
       pp. I-137). IEEE.
    .. [5] Plyer, A., Le Besnerais, G., & Champagnat,
       F. (2016). Massively parallel Lucas Kanade optical flow for
       real-time video processing applications. Journal of Real-Time
       Image Processing, 11(4), 713-730.

    Examples
    --------
    >>> from matplotlib import pyplot as plt
    >>> import pyimof
    >>> I0, I1 = pyimof.data.yosemite()
    >>> u, v = pyimof.solvers.ilk(I0, I1)
    >>> pyimof.display.plot(u, v)
    >>> plt.show()

    """

    solver = partial(_ilk, rad=rad, nwarp=nwarp, gaussian=gaussian,
                     prefilter=prefilter)

    return coarse_to_fine(I0, I1, solver)
