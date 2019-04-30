import pyimof

I0, I1 = pyimof.data.yosemite

u, v = pyimof.solvers.tvl1(I0, I1)
