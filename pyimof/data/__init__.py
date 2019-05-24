"""This module gives access to the two-frames gray scale training
sequences of the Middlebury optical flow dataset [1]_.

.. [1] Baker, S., Scharstein, D., Lewis, J. P., Roth, S., Black,
       M. J., & Szeliski, R. (2011). A database and evaluation
       methodology for optical flow. International Journal of Computer
       Vision, 92(1), 1-31.

"""

from functools import partial
import os
import imageio

__all__ = ['beanbags',
           'dimetrodon',
           'dogdance',
           'grove2',
           'grove3',
           'hydrangea',
           'minicooper',
           'rubberwhale',
           'urban2',
           'urban3',
           'venus',
           'walking']


_data_path = os.path.dirname(__file__)


def _load_seq(seqname):
    dirname = os.path.join(_data_path, seqname)
    return [imageio.imread(os.path.join(dirname, f))
            for f in sorted(os.listdir(dirname))]


beanbags = partial(_load_seq, seqname='Beanbags')
dimetrodon = partial(_load_seq, seqname='Dimetrodon')
dogdance = partial(_load_seq, seqname='DogDance')
grove2 = partial(_load_seq, seqname='Grove2')
grove3 = partial(_load_seq, seqname='Grove3')
hydrangea = partial(_load_seq, seqname='Hydrangea')
minicooper = partial(_load_seq, seqname='MiniCooper')
rubberwhale = partial(_load_seq, seqname='RubberWhale')
urban2 = partial(_load_seq, seqname='Urban2')
urban3 = partial(_load_seq, seqname='Urban3')
venus = partial(_load_seq, seqname='Venus')
walking = partial(_load_seq, seqname='Walking')
