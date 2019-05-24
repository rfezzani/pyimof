Overview
========

   Pyimof actually only supports gray scale images in its implemented
   methods.

Implemented methods
-------------------

Optical flow algorithms can mainly be classified into

- **global methods** based on pixel-wise matching costs with
  regularization constraints, *ie* based on the `Horn & Schunck`_
  paradigm.
- **local methods** based on local window matching costs, *ie* based
  on the `Lucas-Kanade`_ paradigm.

Pyimof provides one implementation for each of these class of methods:

- **TV-L1**: A popular algorithm introduced by Zack *et al.* [1]_,
  improved in [2]_ and detailed in [3]_ (See :func:`pyimof.solvers.tvl1`).
- **Iterative Lucas-kanade**: A fast and robust algorithm developped
  by Le Besnerais and Champagnat [4]_ and improved in [5]_ (See
  :func:`pyimof.solvers.ilk`).

These two algorithms have been selected for theire relative
speed. Efficient GPU implementations for both of them have been
developped but it is not planned to port Pyimof on this platform.

Datasets
--------

Multiple datasets for the evaluation of optical flow algorithms have
been developped (for example Middlebury_, `MPI-Sintel`_ and `Flying
Chairs`_).  The two-frame grayscale **Middlebury** training dataset
[6]_ is accessible via the :mod:`pyimof.data` module functions to ease
testing.

IO
---

Estimated vector fields can be saved and loaded in the *.flo* file
format using the :mod:`pyimof.io` module.

Visualization
-------------

Visualizing optical flow is made easy using the :mod:`pyimof.display` module

- the :func:`pyimof.display.plot` function applies a colormap
  (preferably circular 😉) to the optical flow according to its
  direction and magnitude. An optional color-wheel showing the color
  code can also be displayed to ease resulting image understanding.
- the :func:`pyimof.display.quiver` function draws a quiver plot with
  multiple option for coloring the arrows and displaying a background
  image.

Moreover, Pyimof gives access to a Matplotlib colormap inspired by the
popular color code used by the Middlebury evaluation site for
displaying algorithms' results.

References
**********

.. [1] Zach, C., Pock, T., & Bischof, H. (2007, September). A
      duality based approach for realtime TV-L 1 optical flow. In Joint
      pattern recognition symposium (pp. 214-223). Springer, Berlin,
      Heidelberg.
.. [2] Wedel, A., Pock, T., Zach, C., Bischof, H., & Cremers,
       D. (2009). An improved algorithm for TV-L 1 optical flow. In
       Statistical and geometrical approaches to visual motion analysis
       (pp. 23-45). Springer, Berlin, Heidelberg.
.. [3] Pérez, J. S., Meinhardt-Llopis, E., & Facciolo,
       G. (2013). TV-L1 optical flow estimation. Image Processing On
       Line, 2013, 137-150.
.. [4] Le Besnerais, G., & Champagnat, F. (2005, September). Dense
       optical flow by iterative local window registration. In IEEE
       International Conference on Image Processing 2005 (Vol. 1,
       pp. I-137). IEEE.
.. [5] Plyer, A., Le Besnerais, G., & Champagnat,
       F. (2016). Massively parallel Lucas Kanade optical flow for
       real-time video processing applications. Journal of Real-Time
       Image Processing, 11(4), 713-730.
.. [6] Baker, S., Scharstein, D., Lewis, J. P., Roth, S., Black, M. J., &
       Szeliski, R. (2011). A database and evaluation methodology for optical
       flow. International Journal of Computer Vision, 92(1), 1-31.

.. _Middlebury: http://vision.middlebury.edu/flow/
.. _`Horn & Schunck`: https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method
.. _`Lucas-Kanade`: https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
.. _`MPI-Sintel`: http://sintel.is.tue.mpg.de/
.. _`Flying Chairs`: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
