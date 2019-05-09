from time import time
import matplotlib.pyplot as plt
import pyimof

fig = plt.figure(figsize=((8, 7)))
ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()

I0, I1 = pyimof.data.minicooper

t0 = time()
u, v = pyimof.solvers.tvl1(I0, I1)
t1 = time()

pyimof.display.quiver(u, v, img=I0, ax=ax1)

img = pyimof.display.flow_to_color(u, v)

ax2.imshow(img)
ax2.set_axis_off()

print("TV-L1 processing time: {:02f}sec".format(t1-t0))

t0 = time()
u, v = pyimof.solvers.ilk(I0, I1)
t1 = time()

pyimof.display.quiver(u, v, img=None, ax=ax3)

img = pyimof.display.flow_to_middlebury(u, v)

ax4.imshow(img)
ax4.set_axis_off()

print("ILK processing time: {:02f}sec".format(t1-t0))

plt.tight_layout()
plt.show()
