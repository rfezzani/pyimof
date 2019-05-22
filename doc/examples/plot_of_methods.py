# coding: utf-8
"""
TV-L1 vs iLK
============

Comparison the TV-L1 and iLK methods for the optical flow estimation.

"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
import pyimof


# --- Load data

I0, I1 = pyimof.data.hydrangea()

fig = plt.figure(figsize=((8, 7)))
ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()
cmap = 'middlebury'
plt.tight_layout()

# --- TV-L1

t0 = time()
u, v = pyimof.solvers.tvl1(I0, I1)
t1 = time()

norm = np.sqrt(u*u + v*v)

pyimof.display.quiver(u, v, c=norm, bg=I0, ax=ax1, bg_cmap='gray',
                      vec_cmap='jet')
pyimof.display.plot(u, v, ax=ax2, cmap=cmap)

print("TV-L1 processing time: {:02f}sec".format(t1-t0))

# --- iLK

t0 = time()
u, v = pyimof.solvers.ilk(I0, I1)
t1 = time()

norm = np.sqrt(u*u + v*v)

pyimof.display.quiver(u, v, c=norm, bg=norm, ax=ax3, vec_cmap='Greys')
pyimof.display.plot(u, v, ax=ax4, cmap=cmap)

print("ILK processing time: {:02f}sec".format(t1-t0))

plt.show()
