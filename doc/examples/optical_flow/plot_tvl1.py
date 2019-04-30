import pyimof

I0, I1 = pyimof.data.yosemite

print(I0.shape)

u, v = pyimof.solvers.tvl1(I0, I1)
