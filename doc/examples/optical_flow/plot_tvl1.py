import numpy as np
import pylab as P
import pyimof

I0, I1 = pyimof.data.yosemite

u, v = pyimof.solvers.tvl1(I0, I1)

fig = P.figure()
ax1, ax2 = fig.subplots(2, 1)
ax2.imshow(np.sqrt(u*u+v*v))
ax2.set_axis_off()

img = pyimof.util.flow_to_color(u, v)

ax1.imshow(img)
ax1.set_axis_off()

P.tight_layout()
P.show()
