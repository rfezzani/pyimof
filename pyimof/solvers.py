"""Collection of available optical flow algorithms.

"""

import pylab as P
from functools import partial
import numpy as np
from .util import warp, coarse_to_fine, forward_diff, div


def _tvl1(I0, I1, u0, v0, dt, lambda_, tau, nwarp, niter):

    nl, nc = I0.shape
    y, x = np.meshgrid(np.arange(nl)-0.5, np.arange(nc)-0.5, indexing='ij')

    f0 = lambda_*tau
    f1 = dt/tau

    # u = np.zeros_like(u0)
    # v = np.zeros_like(u0)

    u = u0.copy()
    v = v0.copy()

    pu1 = np.zeros_like(u0)
    pu2 = np.zeros_like(u0)
    pv1 = np.zeros_like(u0)
    pv2 = np.zeros_like(u0)

    for _ in range(nwarp):
        wI1 = warp(I1, u0, v0, x, y)
        Iy, Ix = np.gradient(wI1)
        NI = Ix*Ix + Iy*Iy + 1e-12

        # fig = P.figure()
        # ax1, ax2, ax3 = fig.subplots(1, 3)
        # ax1.imshow(wI1, cmap='gray')
        # ax2.imshow(NI)
        # ax3.imshow(np.sqrt(u0*u0+v0*v0))

        rho_0 = wI1 - I0 - u0*Ix - v0*Iy

        for __ in range(niter):

            # Data term

            rho = rho_0 + u*Ix + v*Iy

            idx = abs(rho) <= f0*NI

            u_ = u.copy()
            v_ = v.copy()

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
            Q = 1 + np.sqrt(ux*ux + uy*uy)

            pu1 += ux
            pu1 /= Q
            pu2 += uy
            pu2 /= Q

            v = v_ - tau*div(pv1, pv2)

            vx, vy = forward_diff(v)
            vx *= f1
            vy *= f1
            Q = 1 + np.sqrt(vx*vx + vy*vy)

            pv1 += vx
            pv1 /= Q
            pv2 += vy
            pv2 /= Q

        u0, v0 = u.copy(), v.copy()

    return u0, v0


def tvl1(I0, I1, dt=0.15, lambda_=0.05, tau=0.3, nwarp=5, niter=10):
    """Coarse to fine TV-L1 optical flow estimator.

    """

    solver = partial(_tvl1, dt=dt, lambda_=lambda_,
                     tau=tau, nwarp=nwarp, niter=niter)

    return coarse_to_fine(I0, I1, solver)
