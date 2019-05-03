from time import time
import matplotlib.pyplot as plt
import pyimof

I0, I1 = pyimof.data.walking

t0 = time()
u, v = pyimof.solvers.tvl1(I0, I1)
t1 = time()

print("Processing time: {:02f}sec".format(t1-t0))

fig = plt.figure(figsize=((5, 9)))
ax1, ax2 = fig.subplots(2, 1)

pyimof.display.quiver(u, v, ax=ax1)

img = pyimof.display.flow_to_color(u, v)

ax2.imshow(img)
ax2.set_axis_off()

plt.tight_layout()
plt.show()
