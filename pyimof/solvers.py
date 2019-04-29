"""Collection of available optical flow algorithms.

"""

from functools import partial
import numpy as np
from .util import warp, coarse_to_fine


def forward_diff(p):
    """Forward difference scheme

    """
    p_x = p.copy()
    p_x[:, :-1] -= p[:, 1:]
    p_x[:, -1] = p_x[:, -2]

    p_y = p.copy()
    p_y[:-1, :] -= p[1:, :]
    p_y[-1, :] = p_y[-2, :]

    return p_x, p_y


def div(p1, p2):
    """Divergence of P=(p1, p2) using backward differece scheme.

    """
    p1_x = p1.copy()
    p1_x[:, 1:] -= p1[:, :-1]
    p1_x[:, 0] = p1_x[:, 1]

    p2_y = p2.copy()
    p2_y[1:, :] -= p2[:-1, :]
    p2_y[0, :] = p2_y[1, :]

    return p1_x + p2_y


def _tvl1(I0, I1, u0, v0, dt=0.1, lambda_=10, tau=10, nwarp=5, niter=100):

    nl, nc = I0.shape
    y, x = np.meshgrid(np.arange(nl), np.arange(nc), indexing='ij')

    u = u0.copy()
    v = v0.copy()

    f = lambda_*tau

    pu1 = np.zeros_like(u0)
    pu2 = np.zeros_like(u0)
    pv1 = np.zeros_like(u0)
    pv2 = np.zeros_like(u0)

    u_ = np.empty_like(u)
    v_ = np.empty_like(v)

    for _ in range(nwarp):
        wI1 = warp(I1, u0, v0, x, y)
        Iy, Ix = np.gradient(wI1)
        NI = Ix*Ix + Iy*Iy

        for __ in range(niter):

            # Data term

            rho = wI1 + (u - u0) * Ix + (v - v0) * Iy - I0

            idx = abs(rho) <= NI

            u_[idx] = -rho[idx]*Ix[idx]/NI[idx]
            v_[idx] = -rho[idx]*Iy[idx]/NI[idx]

            idx = ~idx
            srho = -f*np.sign(rho[idx])
            u_[idx] = srho*Ix[idx]
            v_[idx] = srho*Iy[idx]

            # Regularization term

            u = u_ - tau*div(pu1, pu2)
            v = v_ - tau*div(pv1, pv2)

            # update dual term pu and pv

            ux, uy = forward_diff(u)
            Nu = 1/(1 + dt/tau*(ux*ux + uy*uy))

            pu1 += dt/tau*ux
            pu1 *= Nu
            pu2 += dt/tau*uy
            pu2 *= Nu

            vx, vy = forward_diff(v)
            Nv = 1/(1 + dt/tau*(vx*vx + vy*vy))

            pv1 += dt/tau*vx
            pv1 *= Nv
            pv2 += dt/tau*vy
            pv2 *= Nv

        u0, v0 = u, v

    return u, v


def tvl1(I0, I1, niter=100):
    """Coarse to fine TV-L1 optical flow estimator.

    """

    solver = partial(_tvl1, niter=niter)

    return coarse_to_fine(I0, I1, solver)
