import os
import imageio

data_path = os.path.abspath(os.path.dirname(__file__))

yosemite = [imageio.imread(os.path.join(data_path, 'Yosemite', f))
            for f in sorted(os.listdir(os.path.join(data_path, 'Yosemite')))]
