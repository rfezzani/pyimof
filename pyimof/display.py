import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import skimage


def flow_to_color(u, v):
    nl, nc = u.shape

    flow = np.empty_like(u, dtype=np.complex)
    flow.real = u
    flow.imag = v
    magnitude, angle = np.absolute(flow), np.angle(flow)

    # Map the magnitude between 0 and 1
    magnitude -= magnitude.min()
    magnitude /= magnitude.max()

    # Set hue according to flow direction
    angle *= 0.5/np.pi

    # Create the corresponding HSV image
    img = np.ones((nl, nc, 3))
    img[..., 0] = angle
    img[..., 2] = magnitude

    return skimage.color.hsv2rgb(img)


def computeColor(u, v):
    """Color codes flow field (u, v)

    """
    nanIdx = np.logical_or(np.isnan(u), np.isnan(v))
    nl, nc = u.shape
    u[nanIdx] = 0.
    v[nanIdx] = 0.

    colorwheel = makeColorwheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u)/np.pi

    fk = ((a+1) / 2) * (ncols-1)  # -1~1 maped to 0->ncols-1
    k0 = np.int32(fk)  # 0, 1, 2, ..., ncols-1
    k1 = k0+1
    k1[k1 == ncols] = 0  # 1
    f = fk - k0

    img = np.zeros((nl, nc, 3))
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[:, :, i] = np.uint8(255*col*(1-nanIdx))
    return img/255


def makeColorwheel():
    """Color encoding scheme

    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm

    """

    col_len = [0, 15, 6, 4, 11, 13, 6]
    col_range = np.cumsum(col_len)
    ncol = col_range[-1]
    colorWheel = np.zeros((ncol, 3))

    for idx, (i0, i1, l) in enumerate(zip(col_range[:-1],
                                          col_range[1:],
                                          col_len[1:])):
        j0 = (idx//2) % 3
        j1 = (j0+1) % 3
        if idx & 1:
            colorWheel[i0:i1, j0] = 255 - np.floor(255*np.arange(l)/l)
            colorWheel[i0:i1, j1] = 255
        else:
            colorWheel[i0:i1, j0] = 255
            colorWheel[i0:i1, j1] = np.floor(255*np.arange(l)/l)

    return colorWheel


def flow_to_middlebury(u, v, maxFlow=0):
    UNKNOWN_FLOW_THRESH = 1e9
    eps = np.finfo(float).eps

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999
    maxrad = -1

    # fix unknown flow
    idxUnknown = (np.absolute(u) >= UNKNOWN_FLOW_THRESH) + \
                 (np.absolute(v) >= UNKNOWN_FLOW_THRESH)
    u[idxUnknown] = 0.
    v[idxUnknown] = 0.

    maxu = max(maxu, u.max())
    minu = min(minu, u.min())

    maxv = max(maxv, v.max())
    minv = min(minv, v.min())

    rad = np.sqrt(u**2+v**2)
    maxrad = max(maxrad, rad.max()) + eps

    print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f' % (
        maxrad, minu, maxu, minv, maxv))

    if maxFlow > 0:
        maxrad = maxFlow

    u /= maxrad
    v /= maxrad

    # compute color
    img = computeColor(u, v)

    # unknown flow
    img[..., 0][idxUnknown] = 0
    img[..., 1][idxUnknown] = 0
    img[..., 2][idxUnknown] = 0

    return img


def gradPatern(truerange, height=151., width=151., expend=1.04):
    """Create a graduated patern

    """
    Range = truerange * expend

    if Range < 2:
        g = 100
    elif Range < 4:
        g = 50
    elif Range < 6:
        g = 30
    elif Range < 11:
        g = 20
    elif Range < 21:
        g = 10
    elif Range < 31:
        g = 6
    else:
        g = 3

    height = 2*np.round(Range)*g
    width = height

    s2 = np.round(height/2)

    y, x = np.mgrid[1:width+1, 1:height+1]

    u = x*Range/s2 - Range
    v = y*Range/s2 - Range

    img = computeColor(u/truerange, v/truerange)

    # On cree la mire
    img[s2, ...] = 0
    img[:, s2, :] = 0

    # On la gradue
    if Range >= 2:
        img[s2+1, g::g, :] = 0
        img[s2-1, g::g, :] = 0
        img[g::g, s2+1, :] = 0
        img[g::g, s2-1, :] = 0
    if Range > 5:
        G = g*(np.round(Range) % 5)
        img[s2+2, G::5*g, :] = 0
        img[s2-2, G::5*g, :] = 0
        img[G::5*g, s2+2, :] = 0
        img[G::5*g, s2-2, :] = 0

    if Range > 10:
        G = g*(np.round(Range) % 10)
        img[s2+3, G::10*g, :] = 0
        img[s2-3, G::10*g, :] = 0
        img[G::10*g, s2+3, :] = 0
        img[G::10*g, s2-3, :] = 0
    return img


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
        uy, ux = np.gradient(u)
        vy, vx = np.gradient(v)
        outp = vx - uy
    else:
        raise ValueError("Unknown method, method must be " +
                         "'circulation','richardson','leastsq' or 'centered'")

    return outp


def vorticityColorMap():
    """Give a well suited colormap to visualize Vorticity using Pylab.

    """
    cdict = {'red':   [(0.0,  1.0, 1.0),
                       (0.2,  0.0, 0.0),
                       (0.4,  0.0, 0.0),
                       (0.5,  1.0, 1.0),
                       (0.6,  0.0, 0.0),
                       (0.7,  0.0, 0.0),
                       (0.8,  1.0, 1.0),
                       (1.0,  1.0, 1.0)],

             'green': [(0.0,  0.0, 0.0),
                       (0.2,  0.0, 0.0),
                       (0.4,  1.0, 1.0),
                       (0.5,  1.0, 1.0),
                       (0.6,  1.0, 1.0),
                       (0.7,  1.0, 1.0),
                       (0.8,  1.0, 1.0),
                       (1.0,  0.0, 0.0)],

             'blue':  [(0.0,  1.0, 1.0),
                       (0.2,  1.0, 1.0),
                       (0.4,  1.0, 1.0),
                       (0.5,  1.0, 1.0),
                       (0.6,  0.0, 0.0),
                       (0.7,  0.0, 0.0),
                       (0.8,  0.0, 0.0),
                       (1.0,  0.0, 0.0)]}

    return plt.cm.colors.LinearSegmentedColormap('vort_cm', cdict)


def figSize(nl, nc, dimBound=6):
    dpi = plt.rcParams['figure.dpi']
    h = float(nl)/dpi
    w = float(nc)/dpi
    maxDim = max(h, w)
    redFact = dimBound/maxDim
    h *= redFact
    w *= redFact
    return w, h


def quiver(u, v, ax=None, sstep=None):
    nl, nc = u.shape

    if ax is None:
        figsize = figSize(nl, nc, 6)
        plt.figure(figsize=figsize, facecolor=(1., 1., 1.))
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

    if sstep is None:
        sstep = max(nl//50, nc//50)

    norm = np.sqrt(u**2+v**2)

    b = 25
    a = np.minimum(nc % b, nl % b)
    xmin = a-b
    xmax = nc - xmin
    ymin = a-b
    ymax = nl - ymin

    ax.imshow(norm)

    x, y = np.mgrid[:nl, :nc]
    ax.quiver(y[::sstep, ::sstep], x[::sstep, ::sstep],
              u[::sstep, ::sstep], -v[::sstep, ::sstep],
              units='dots')
    ax.axis([xmin, xmax, ymax, ymin])
    ax.set_axis_off()


def colorCodeShow(u, v, figsize=None):
    if figsize is None:
        nl, nc = u.shape
        figsize = figSize(nl, nc, 6)
    flow = np.zeros((nl, nc, 2))
    flow[..., 0] = u
    flow[..., 1] = v
    img = flow_to_color(flow)
    plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    plt.imshow(img)


def normShow(u, v, figsize=None):
    if figsize is None:
        nl, nc = u.shape
        figsize = figSize(nl, nc, 6)
    norm = np.sqrt(u**2+v**2)
    plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    plt.imshow(norm)


def vortShow(u, v, method='centered', tresh=0.4, norm=None, figsize=None):
    vort = vorticity(u, v, method, DeltaX=1., DeltaY=1.)
    if norm is None:
        norm = np.sqrt(u**2+v**2)
    if figsize is None:
        nl, nc = u.shape
        figsize = figSize(nl, nc, 6)
    vort[norm < tresh] = 0.
    cm = vorticityColorMap()

    plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    plt.imshow(vort, cmap=cm, vmin=-0.2, vmax=0.2)


def pctileTreshold(u, v, pctil=99):

    nl, nc = u.shape
    norm = np.sqrt(u**2+v**2)

    if (100. - pctil):
        max_norm = np.sort(norm, axis=None)[np.floor(nl*nc*pctil/100)]
        idx = np.where(norm > max_norm)
        u[idx] /= norm[idx]
        u[idx] *= max_norm
        v[idx] /= norm[idx]
        v[idx] *= max_norm

    return u, v
