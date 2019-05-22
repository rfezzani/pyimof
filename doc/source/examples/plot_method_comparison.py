# coding: utf-8
"""
TV-L1 vs iLK
============

Comparison the TV-L1 and iLK methods for the optical flow estimation.

"""

from time import time
import matplotlib.pyplot as plt
import pyimof


seq_list = pyimof.data.__all__
seq_count = len(seq_list)

fig = plt.figure(figsize=((9, 2*seq_count)))
ax_array = fig.subplots(seq_count, 3)
ax_array[0, 0].set_title("Reference image")
ax_array[0, 1].set_title("TV-L1 result")
ax_array[0, 2].set_title("iLK result")

# --- Loop over available sequences

for name, (ax0, ax1, ax2) in zip(seq_list, ax_array):

    title = name.capitalize()

    print(f"Processing the {title} sequence")

    # --- Load data
    I0, I1 = pyimof.data.__dict__[name]()

    ax0.imshow(I0, cmap='gray')
    ax0.set_ylabel(title)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # --- Run TV-L1

    t0 = time()
    u, v = pyimof.solvers.tvl1(I0, I1)
    t1 = time()

    pyimof.display.plot(u, v, ax=ax1, colorwheel=False)

    print("\tTV-L1 processing time: {:02f}sec".format(t1-t0))

    # --- Run iLK

    t0 = time()
    u, v = pyimof.solvers.ilk(I0, I1)
    t1 = time()

    pyimof.display.plot(u, v, ax=ax2, colorwheel=False)

    print("\tILK processing time: {:02f}sec".format(t1-t0))

fig.tight_layout()

plt.show()
