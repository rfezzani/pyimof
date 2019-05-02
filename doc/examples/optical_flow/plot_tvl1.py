import pylab as P
import pyimof

I0, I1 = pyimof.data.yosemite

u, v = pyimof.solvers.tvl1(I0, I1)

P.imshow(u*u+v*v)
P.show()
