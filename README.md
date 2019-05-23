[![Documentation Status](https://readthedocs.org/projects/pyimof/badge/?version=latest)](https://pyimof.readthedocs.io/en/latest/?badge=latest)

Pyimof (for **Py**thon **im**age **o**ptical **f**low) is a pure
python package for dense [optical
flow](https://en.wikipedia.org/wiki/Optical_flow) estimation. Please
visit the [documentation](https://pyimof.readthedocs.io/en/latest/)
where some examples are provided in the
[gallery](https://pyimof.readthedocs.io/en/latest/examples/index.html)

## Quick Example

Using Pyimov is as easy as

```python
from matplotlib import pyplot as plt
import pyimof
I0, I1 = pyimof.data.hydrangea()
u, v = pyimof.solvers.tvl1(I0, I1)
pyimof.display.plot(u, v)
plt.show()
```

to obtain

![Hydrangea](doc/source/_static/hydrangea.png)
