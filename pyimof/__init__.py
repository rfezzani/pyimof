# coding: utf-8
"""A python package for optical flow estimation and visualization.

"""

import matplotlib.pyplot as plt
from . import data
from . import io
from . import solvers
from . import display


# Registration of the middlebury color map
plt.cm.register_cmap(
    'middlebury',
    cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
        'middlebury', display._middlebury()).reversed())

# Registration of reversed version of the middlebury color map
plt.cm.register_cmap(
    'middlebury_r',
    cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
        'middlebury_r', display._middlebury()))
