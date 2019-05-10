import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import skimage


def _get_color_code():
    """Color encoding scheme

    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm

    """

    col_len = [0, 15, 6, 4, 11, 13, 6]
    col_range = np.cumsum(col_len)
    ncol = col_range[-1]
    cmap = np.zeros((3, ncol))

    for idx, (i0, i1, l) in enumerate(zip(col_range[:-1],
                                          col_range[1:],
                                          col_len[1:])):
        j0 = (idx//2) % 3
        j1 = (j0+1) % 3
        if idx & 1:
            cmap[j0, i0:i1] = 1 - np.arange(l)/l
            cmap[j1, i0:i1] = 1
        else:
            cmap[j0, i0:i1] = 1
            cmap[j1, i0:i1] = np.arange(l)/l

    return cmap.T


def flow_to_color(u, v, cmap='middlebury', invert=True):
    nl, nc = u.shape

    flow = np.empty_like(u, dtype=np.complex)
    flow.real = u
    flow.imag = v
    if invert:
        flow = -flow
    magnitude = np.absolute(flow)

    max_val = magnitude.max()
    flow /= max_val
    angle = np.angle(flow)

    if cmap is None:

        # Map the magnitude between 0 and 1
        magnitude -= magnitude.min()
        magnitude /= max_val

        # Set hue according to flow direction
        angle *= 0.5/np.pi

        # Create the corresponding HSV image
        hsv = np.ones((nl, nc, 3))
        hsv[..., 0] = angle
        hsv[..., 2] = magnitude

        img = skimage.color.hsv2rgb(hsv)
    else:
        norm = plt.Normalize()
        lut = plt.cm.get_cmap(cmap)
        img = lut(norm(angle))
        img[..., 3] = (magnitude/max_val)**0.7

    return img


def color_wheel(u, v, nr=50, ntheta=1025, invert=True):
    rad = np.sqrt(u*u + v*v)
    r_max, r_min = rad.max(), rad.min()
    r, t = np.mgrid[r_min:r_max:nr*1j, 0:2*np.pi:ntheta*1j]
    vals = np.mod(t.copy(), 2*np.pi)
    if invert:
        vals = -vals
    return t, r, vals


def vorticity(u, v, method='leastsq', DeltaX=1., DeltaY=1.):
    """outp = vorticity(u,v,method='centered',DeltaX=1.,DeltaY=1.)

    compute the vorticity of a 2D flow field (u,v) computed on a
    uniformly spaced grid.

    u: horizontal componant of the flow
    v: vertical component of the flow
    method (default = 'leastsq'): specify the used method
           should be 'circulation', 'richardson', 'leastsq' or 'centered'.
    DeltaX, DeltaY: respectively the horizontal an vertical grid step.
           (Default = 1.)

    """

    if method == 'circulation':
        Dx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])/(8*DeltaX)
        Dy = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])/(8*DeltaY)
        Vort = ndimage.convolve(v, Dx) + ndimage.convolve(u, Dy)
        outp = -np.real(Vort)
        outp[0, :] = 0.
        outp[-1, :] = 0.
        outp[:, -1] = 0.
        outp[:, 0] = 0.

    elif method == 'richardson':

        Dx = np.array([-1, 8, 0, -8, 1])/(12*DeltaX)
        Dy = np.array([-1, 8, 0, -8, 1])/(12*DeltaY)
        outp = (ndimage.convolve1d(v, Dx, axis=1) -
                ndimage.convolve1d(u, Dy, axis=0))
        outp[:2, :] = 0.
        outp[-2:, :] = 0.
        outp[:, :2] = 0.
        outp[:, -2:] = 0.

    elif method == 'leastsq':

        Dx = np.array([2, 1, 0, -1, -2])/(10*DeltaX)
        Dy = np.array([2, 1, 0, -1, -2])/(10*DeltaY)
        outp = (ndimage.convolve1d(v, Dx, axis=1) -
                ndimage.convolve1d(u, Dy, axis=0))
        outp[:2, :] = 0.
        outp[-2:, :] = 0.
        outp[:, :2] = 0.
        outp[:, -2:] = 0.

    elif method == 'centered':
        uy, _ = np.gradient(u)
        _, vx = np.gradient(v)
        outp = vx - uy
    else:
        raise ValueError("Unknown method, method must be " +
                         "'circulation','richardson','leastsq' or 'centered'")

    return outp


def get_tight_figsize(nl, nc, dimBound=6):
    dpi = plt.rcParams['figure.dpi']
    h = float(nl)/dpi
    w = float(nc)/dpi
    maxDim = max(h, w)
    redFact = dimBound/maxDim
    h *= redFact
    w *= redFact
    return w, h


def quiver(u, v, img=None, ax=None, sstep=None):
    nl, nc = u.shape

    if ax is None:
        figsize = get_tight_figsize(nl, nc)
        plt.figure(figsize=figsize, facecolor=(1., 1., 1.))
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

    if sstep is None:
        sstep = max(nl//50, nc//50)

    norm = np.sqrt(u**2+v**2)
    img_cm = 'gray'
    vec_cm = 'jet'
    if img is None:
        img = norm
        img_cm = 'viridis'
        vec_cm = 'Greys'

    ax.imshow(img, cmap=img_cm)

    y, x = np.mgrid[:nl:sstep, :nc:sstep]
    u_ = u[::sstep, ::sstep]
    v_ = v[::sstep, ::sstep]
    idx = np.logical_and(np.logical_and(x+u_ >= 0, x+u_ <= nc-1),
                         np.logical_and(y+v_ >= 0, y+v_ <= nl-1))
    norm = norm[::sstep, ::sstep]
    ax.quiver(x[idx], y[idx], u_[idx], v_[idx], norm[idx], units='dots',
              angles='xy', scale_units='xy', cmap=vec_cm)
    ax.set_axis_off()


def plot(u, v, ax=None, cmap='middlebury', invert=True, colorwheel=True):
    nl, nc = u.shape

    if ax is None:
        figsize = get_tight_figsize(nl, nc)
        plt.figure(figsize=figsize, facecolor=(1., 1., 1.))
        ax = plt.axes([0, 0, 1, 1], frameon=False)

    flow = np.empty_like(u, dtype=np.complex)
    flow.real = u
    flow.imag = v
    if invert:
        flow = -flow
    magnitude = np.absolute(flow)

    max_val = magnitude.max()
    flow /= max_val
    angle = np.angle(flow)

    norm = plt.Normalize()
    lut = plt.cm.get_cmap(cmap)
    img = lut(norm(angle))
    img[..., 3] = (magnitude/max_val)**0.7

    ax.imshow(img)
    ax.set_axis_off()

    if colorwheel:
        bbox = ax.get_position()
        w, h = bbox.width, bbox.height
        X0, Y0 = bbox.x0, bbox.y0

        s = max(w*0.2, h*0.2)
        x0, y0 = X0+0.1*w, Y0+0.7*h

        fig = ax.get_figure()
        ax2 = fig.add_axes([x0, y0, s, s], polar=True)
        t, r, vals = color_wheel(u, v)
        ax2.pcolormesh(t, r, vals, cmap=cmap)
