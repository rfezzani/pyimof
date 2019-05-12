"""Collection of available optical flow algorithms.

"""

from functools import partial
import numpy as np
from scipy import ndimage as ndi
from .util import warp, coarse_to_fine, central_diff, forward_diff, div


def _tvl1(I0, I1, u0, v0, dt, lambda_, tau, nwarp, niter, tol):

    nl, nc = I0.shape
    y, x = np.meshgrid(np.arange(nl), np.arange(nc), indexing='ij')

    f0 = lambda_*tau
    f1 = dt/tau

    u = u0.copy()
    v = v0.copy()

    pu1 = np.zeros_like(u0)
    pu2 = np.zeros_like(u0)
    pv1 = np.zeros_like(u0)
    pv2 = np.zeros_like(u0)

    for _ in range(nwarp):
        wI1 = warp(I1, u0, v0, x, y)
        Ix, Iy = central_diff(wI1)
        NI = Ix*Ix + Iy*Iy
        NI[NI == 0] = 1

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

            for ___ in range(2):

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

        u0 -= u
        v0 -= v
        if (u0*u0+v0*v0).sum()/(u.size) < tol:
            break
        else:
            u0, v0 = u.copy(), v.copy()

    return u, v


def tvl1(I0, I1, dt=0.2, lambda_=15, tau=0.3, nwarp=5, niter=10, tol=1e-4):
    """Coarse to fine TV-L1 optical flow estimator.

    """

    solver = partial(_tvl1, dt=dt, lambda_=lambda_,
                     tau=tau, nwarp=nwarp, niter=niter, tol=tol)

    return coarse_to_fine(I0, I1, solver)


def _ilk(I0, I1, u0, v0, rad, nwarp):

    nl, nc = I0.shape
    y, x = np.meshgrid(np.arange(nl), np.arange(nc), indexing='ij')

    size = 2*rad+1

    u = u0.copy()
    v = v0.copy()

    for _ in range(nwarp):
        wI1 = warp(I1, u, v, x, y)
        Ix, Iy = central_diff(wI1)
        It = wI1 - I0 - u*Ix - v*Iy

        J11 = Ix*Ix
        J12 = Ix*Iy
        J22 = Iy*Iy
        J13 = Ix*It
        J23 = Iy*It

        ndi.uniform_filter(J11, size=size, output=J11, mode='mirror')
        ndi.uniform_filter(J12, size=size, output=J12, mode='mirror')
        ndi.uniform_filter(J22, size=size, output=J22, mode='mirror')
        ndi.uniform_filter(J13, size=size, output=J13, mode='mirror')
        ndi.uniform_filter(J23, size=size, output=J23, mode='mirror')

        detA = -(J11*J22 - J12*J12)
        idx = abs(detA) < 1e-14
        detA[idx] = 1

        u = (J13*J22 - J12*J23)/detA
        v = (J23*J11 - J12*J13)/detA

        u[idx] = 0
        v[idx] = 0

    return u, v


def ilk(I0, I1, rad=7, nwarp=10):

    solver = partial(_ilk, rad=rad, nwarp=nwarp)

    return coarse_to_fine(I0, I1, solver)
