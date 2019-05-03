import os
import imageio

data_path = os.path.abspath(os.path.dirname(__file__))

yosemite = [imageio.imread(os.path.join(data_path, 'Yosemite', f))
            for f in sorted(os.listdir(os.path.join(data_path, 'Yosemite')))]

beanbags = [imageio.imread(os.path.join(data_path, 'Beanbags', f))
            for f in sorted(os.listdir(os.path.join(data_path, 'Beanbags')))]

dogdance = [imageio.imread(os.path.join(data_path, 'DogDance', f))
            for f in sorted(os.listdir(os.path.join(data_path, 'DogDance')))]

grove3 = [imageio.imread(os.path.join(data_path, 'Grove3', f))
          for f in sorted(os.listdir(os.path.join(data_path, 'Grove3')))]

urban2 = [imageio.imread(os.path.join(data_path, 'Urban2', f))
          for f in sorted(os.listdir(os.path.join(data_path, 'Urban2')))]

venus = [imageio.imread(os.path.join(data_path, 'Venus', f))
         for f in sorted(os.listdir(os.path.join(data_path, 'Venus')))]

dimetrodon = [imageio.imread(os.path.join(data_path, 'Dimetrodon', f))
              for f in sorted(os.listdir(os.path.join(data_path,
                                                      'Dimetrodon')))]

grove2 = [imageio.imread(os.path.join(data_path, 'Grove2', f))
          for f in sorted(os.listdir(os.path.join(data_path, 'Grove2')))]

hydrangea = [imageio.imread(os.path.join(data_path, 'Hydrangea', f))
             for f in sorted(os.listdir(os.path.join(data_path, 'Hydrangea')))]

minicooper = [imageio.imread(os.path.join(data_path, 'MiniCooper', f))
              for f in sorted(os.listdir(os.path.join(data_path,
                                                      'MiniCooper')))]

rubberwhale = [imageio.imread(os.path.join(data_path, 'RubberWhale', f))
               for f in sorted(os.listdir(os.path.join(data_path,
                                                       'RubberWhale')))]

urban3 = [imageio.imread(os.path.join(data_path, 'Urban3', f))
          for f in sorted(os.listdir(os.path.join(data_path, 'Urban3')))]

walking = [imageio.imread(os.path.join(data_path, 'Walking', f))
           for f in sorted(os.listdir(os.path.join(data_path, 'Walking')))]
