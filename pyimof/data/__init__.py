from functools import partial
import os
import imageio

data_path = os.path.abspath(os.path.dirname(__file__))


def load_seq(dirname):
    return [imageio.imread(os.path.join(data_path, dirname, f))
            for f in sorted(os.listdir(os.path.join(data_path, dirname)))]


beanbags = partial(load_seq, dirname='Beanbags')
dimetrodon = partial(load_seq, dirname='Dimetrodon')
dogdance = partial(load_seq, dirname='DogDance')
grove2 = partial(load_seq, dirname='Grove2')
grove3 = partial(load_seq, dirname='Grove3')
hydrangea = partial(load_seq, dirname='Hydrangea')
minicooper = partial(load_seq, dirname='MiniCooper')
rubberwhale = partial(load_seq, dirname='RubberWhale')
urban2 = partial(load_seq, dirname='Urban2')
urban3 = partial(load_seq, dirname='Urban3')
venus = partial(load_seq, dirname='Venus')
walking = partial(load_seq, dirname='Walking')
yosemite = partial(load_seq, dirname='Yosemite')
