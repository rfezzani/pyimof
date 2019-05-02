"""Collection of available optical flow algorithms.

"""

from functools import partial
import numpy as np
from .util import warp, coarse_to_fine, forward_diff, div


def _tvl1(I0, I1, u0, v0, dt=0.1, lambda_=50, tau=10, nwarp=5, niter=100):

    nl, nc = I0.shape
    y, x = np.meshgrid(np.arange(nl), np.arange(nc), indexing='ij')

    u = u0.copy()
    v = v0.copy()

    f0 = lambda_*tau
    f1 = -dt/tau

    pu1 = np.zeros_like(u0)
    pu2 = np.zeros_like(u0)
    pv1 = np.zeros_like(u0)
    pv2 = np.zeros_like(u0)

    u_ = np.empty_like(u)
    v_ = np.empty_like(v)

    for _ in range(nwarp):
        wI1 = warp(I1, u0, v0, x, y)
        Iy, Ix = np.gradient(wI1)
        NI = Ix*Ix + Iy*Iy + 1e-12

        rho_0 = wI1 - u0 * Ix - v0 * Iy - I0

        for __ in range(niter):

            # Data term

            rho = rho_0 + u*Ix + v*Iy

            u_ = u.copy()
            v_ = v.copy()

            idx = abs(rho) <= f0*NI

            u_[idx] -= rho[idx]*Ix[idx]/NI[idx]
            v_[idx] -= rho[idx]*Iy[idx]/NI[idx]

            idx = ~idx
            srho = f0*np.sign(rho[idx])
            u_[idx] -= srho*Ix[idx]
            v_[idx] -= srho*Iy[idx]

            # Regularization term

            u = u_ - tau*div(pu1, pu2)

            ux, uy = forward_diff(u)
            ux *= f1
            uy *= f1
            Q = 1 - np.sqrt(ux*ux + uy*uy)

            pu1 += ux
            pu1 /= Q
            pu2 += uy
            pu2 /= Q

            v = v_ - tau*div(pv1, pv2)

            vx, vy = forward_diff(v)
            vx *= f1
            vy *= f1
            Q = 1 - np.sqrt(vx*vx + vy*vy)

            pv1 += vx
            pv1 /= Q
            pv2 += vy
            pv2 /= Q

        u0, v0 = u.copy(), v.copy()

    return u, v


def tvl1(I0, I1, niter=100):
    """Coarse to fine TV-L1 optical flow estimator.

    """

    solver = partial(_tvl1, niter=niter)

    return coarse_to_fine(I0, I1, solver)
