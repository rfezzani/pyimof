.. Pyimof documentation master file, created by
   sphinx-quickstart on Wed May 22 00:10:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Pyimof's documentation!
==================================


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   modules
   examples/index

Introduction
============

Pyimof (for **Py**\ thon **im**\ age **o**\ ptical **f**\ low) is a
pure python package for dense `optical flow`_ estimation. A good
introduction to optical flow techniques can be found here_.

Quick Example
-------------

Using Pyimov is as easy as

>>> from matplotlib import pyplot as plt
>>> import pyimof
>>> I0, I1 = pyimof.data.hydrangea()
>>> u, v = pyimof.solvers.tvl1(I0, I1)
>>> pyimof.display.plot(u, v)
>>> plt.show()

to obtain

.. image:: _static/hydrangea.png

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _`optical flow`: https://en.wikipedia.org/wiki/Optical_flow
.. _here: https://blog.nanonets.com/optical-flow/
