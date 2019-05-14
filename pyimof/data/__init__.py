from functools import partial
import os
import imageio


def load_seq(dirname):
    return [imageio.imread(os.path.join(dirname, f))
            for f in sorted(os.listdir(dirname))]


data_path = os.path.abspath(os.path.dirname(__file__))

beanbags = partial(load_seq, dirname=os.path.join(data_path, 'Beanbags'))
dimetrodon = partial(load_seq, dirname=os.path.join(data_path, 'Dimetrodon'))
dogdance = partial(load_seq, dirname=os.path.join(data_path, 'DogDance'))
grove2 = partial(load_seq, dirname=os.path.join(data_path, 'Grove2'))
grove3 = partial(load_seq, dirname=os.path.join(data_path, 'Grove3'))
hydrangea = partial(load_seq, dirname=os.path.join(data_path, 'Hydrangea'))
minicooper = partial(load_seq, dirname=os.path.join(data_path, 'MiniCooper'))
rubberwhale = partial(load_seq, dirname=os.path.join(data_path, 'RubberWhale'))
urban2 = partial(load_seq, dirname=os.path.join(data_path, 'Urban2'))
urban3 = partial(load_seq, dirname=os.path.join(data_path, 'Urban3'))
venus = partial(load_seq, dirname=os.path.join(data_path, 'Venus'))
walking = partial(load_seq, dirname=os.path.join(data_path, 'Walking'))
yosemite = partial(load_seq, dirname=os.path.join(data_path, 'Yosemite'))
