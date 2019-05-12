import matplotlib.pyplot as plt
from . import data
from . import io
from . import solvers
from . import display


plt.cm.register_cmap(
    'middlebury',
    cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
        'middlebury', display._get_color_code()).reversed())

plt.cm.register_cmap(
    'middlebury_r',
    cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
        'middlebury_r', display._get_color_code()))
