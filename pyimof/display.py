# coding: utf-8
"""Collection of utils and functions for the visualization of vector
fields.

"""

import matplotlib.pyplot as plt
import numpy as np
import skimage


def _middlebury():
    """Compute the colors used to generate the `middlebury` colormap.

    This colormap is inspired by the middlebury evaluation for optical
    flow algorithms [1]_. The RGB values are extracted from the matlab
    code [2]_

    Returns
    -------
    cmap : ~numpy.ndarray
        The colors used to generate the 'middlebury' colormap.

    References
    ----------
    .. [1] http://vision.middlebury.edu/flow/
    .. [2] http://vision.middlebury.edu/flow/code/flow-code-matlab.zip

    """

    col_len = [0, 15, 6, 4, 11, 13, 6]
    col_range = np.cumsum(col_len)
    ncol = col_range[-1]
    cmap = np.zeros((ncol, 3))

    for idx, (i0, i1, l) in enumerate(zip(col_range[:-1],
                                          col_range[1:],
                                          col_len[1:])):
        j0 = (idx//2) % 3
        j1 = (j0+1) % 3
        if idx & 1:
            cmap[i0:i1, j0] = 1 - np.arange(l)/l
            cmap[i0:i1, j1] = 1
        else:
            cmap[i0:i1, j0] = 1
            cmap[i0:i1, j1] = np.arange(l)/l

    return cmap


def flow_to_color(u, v, cmap=None, scale=True):
    """Apply color code to a vector field according to its orientation and
    magnitude.

    Any colormap compatible with matplotlib can be applyed but
    circular colormaps are recommanded ( for example 'huv',
    'twilight', 'twilight_shifted' and the builtin 'middlebury'
    colormaps).

    If cmap is None, the HSV image defined using optical flow
    orientation (hue) and magnitude (saturation) is returned.

    Parameters
    ----------
    u : ~numpy.ndarray
        The horizontal component of the vector field.
    v : ~numpy.ndarray
        The vertical component of the vector field.
    cmap : str (optional)
        The colormap used to color code the input vector field.
    scale : bool (optional)
        whether to scale output saturation according to magnitude.

    Returns
    -------
    img : ~numpy.ndarray
        RGBA image representing the desired color code applyed to the
        vector field.

    """

    flow = (u + 1j*v)
    magnitude = np.absolute(flow)
    angle = np.mod(-np.angle(flow), 2*np.pi)

    # Normalize flow direction
    angle /= (2*np.pi)

    # Map the magnitude between 0 and 1
    magnitude -= magnitude.min()
    magnitude /= magnitude.max()

    if cmap is None:

        # Create the corresponding HSV image
        nl, nc = u.shape
        hsv = np.ones((nl, nc, 3))
        hsv[..., 0] = angle
        hsv[..., 2] = magnitude

        img = skimage.color.hsv2rgb(hsv)
    else:
        lut = plt.cm.get_cmap(cmap)
        img = lut(angle)
        if scale:
            # Scale saturation according to magnitude
            img[..., 3] = magnitude

    return img


def color_wheel(u=None, v=None, nr=50, ntheta=1025):
    """Compute the discretization of a wheel used to describe
    the color code used to display a vector field (u, v).

    If the vector field (u, v) is provided, the radius of the wheel is
    equal to its maximum magnitude. Otherwise (i.e. if any of u and v
    is None), the radius is set to 1.

    Parameters
    ----------
    u : ~numpy.ndarray (optional)
        The horizontal component of the vector field (default: None).
    v : ~numpy.ndarray (optional)
        The vertical component of the vector field (default: None).
    nr : int (optional)
        The number of steps used to discretise the wheel radius.
    ntheta : int (optional)
        The number of steps used to discretise the wheel sectors.

    Returns
    -------
    angle, radius: tuple[~numpy.ndarray]
        The grid discretisation of the wheel sectors and radius.

    """
    max_rad = 1
    if u is not None or v is not None:
        max_rad = np.sqrt(u*u + v*v).max()
    radius, angle = np.mgrid[:max_rad:nr*1j, 0:2*np.pi:ntheta*1j]
    return angle, radius


def get_tight_figsize(I):
    """Computes the matplotlib figure tight size respecting image
    proportions.

    Parameters
    ----------
    I : ~numpy.ndarray
        The image to be displayed.

    Returns
    -------
    w, h : tuple[float]
        The width and height in inch of the desired figure.

    """
    nl, nc = I.shape[:2]
    dpi = plt.rcParams['figure.dpi']
    dimBound = max(plt.rcParams['figure.figsize'])
    h = float(nl)/dpi
    w = float(nc)/dpi
    maxDim = max(h, w)
    redFact = dimBound/maxDim
    h *= redFact
    w *= redFact
    return w, h


def plot(u, v, ax=None, cmap='middlebury', scale=True, colorwheel=True):
    """Plots the color coded vector field.

    Parameters
    ----------
    u : ~numpy.ndarray
        The horizontal component of the vector field.
    v : ~numpy.ndarray
        The vertical component of the vector field.
    ax : ~matplotlib.pyplot.Axes (optional)
        Optional matplotlib axes used to plot the image. If None, the
        image is displayed in a tight figure.
    cmap : str (optional)
        The colormap used to color code the input vector field.
    scale : bool (optional)
        whether to scale output saturation according to magnitude.
    colorwheel : bool (optional)
        whether to display the color wheel describing the images
        colors or not.

    Returns
    -------
    ax : ~matplotlib.pyplot.Axes
        The matplotlib axes where the image is displayed.

    """

    img = flow_to_color(u, v, cmap, scale)

    if ax is None:
        nl, nc = u.shape
        figsize = get_tight_figsize(img)
        plt.figure(figsize=figsize, facecolor=(1., 1., 1.))
        ax = plt.axes([0, 0, 1, 1], frameon=False)

    ax.imshow(img)
    ax.set_axis_off()

    if colorwheel:
        if cmap is None:
            cmap = 'hsv'
        bbox = ax.get_position()
        w, h = bbox.width, bbox.height
        X0, Y0 = bbox.x0, bbox.y0

        x0, y0 = X0+0.01*w, Y0+0.79*h

        fig = ax.get_figure()
        ax2 = fig.add_axes([x0, y0, w*0.2, h*0.2], polar=1)
        angle, rad = color_wheel(u, v)
        ax2.pcolormesh(angle, rad, angle, cmap=cmap)
        ax2.set_xticks([])

    return ax


def quiver(u, v, c=None, bg=None, ax=None, step=None, nvec=50, bg_cmap=None,
           **kwargs):
    """Draws a quiver plot representing a dense vector field.

    Parameters
    ----------
    u : ~numpy.ndarray (with shape m×n)
        The horizontal component of the vector field.
    v : ~numpy.ndarray (with shape m×n)
        The vertical component of the vector field.
    c : ~numpy.ndarray (optional (with shape m×n))
        Values used to color the arrows.
    bg : ~numpy.ndarray (2D or 3D optional)
        Background image.
    ax : ~matplotlib.pyplot.Axes (optional)
        Axes used to plot the image. If None, the image is displayed
        in a tight figure.
    step : int (optional)
        The grid step used to display the vector field. If None, it is
        computed using the nvec parameter.
    nvec : int
        The maximum number of vector over all the grid dimentions. It
        is ignored if the step parameter is not None.
    bg_cmap : str (optional)
        The colormap used to color the background image.

    Notes
    -----
    Any other :func:`matplotlib.pyplot.quiver` valid keyword can be
    used, knowing that some are fixed

    - units = 'dots'
    - angles = 'xy'
    - scale = 'xy'

    Returns
    -------
    ax : ~matplotlib.pyplot.Axes
        The matplotlib axes where the vector field is displayed.

    """

    if ax is None:
        figsize = get_tight_figsize(u)
        plt.figure(figsize=figsize, facecolor=(1., 1., 1.))
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

    nl, nc = u.shape

    if step is None:
        step = max(nl//nvec, nc//nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]
    idx = np.logical_and(np.logical_and(x+u_ >= 0, x+u_ <= nc-1),
                         np.logical_and(y+v_ >= 0, y+v_ <= nl-1))

    if bg is not None:
        ax.imshow(bg, cmap=bg_cmap)
    else:
        ax.axis([0, nc, nl, 0])
        ax.set_aspect('equal')

    args = [x[idx], y[idx], u_[idx], v_[idx]]
    if c is not None:
        args.append(c[::step, ::step][idx])

    kwargs['units'] = 'dots'
    kwargs['angles'] = 'xy'
    kwargs['scale_units'] = 'xy'

    ax.quiver(*args, **kwargs)
    ax.set_axis_off()

    return ax
