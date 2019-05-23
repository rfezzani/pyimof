from setuptools import setup, find_packages

setup(
    name='pyimof',
    version='0.0.0',
    description="Python package for optical flow estimation",
    package_dir={'pyimof': 'pyimof'},
    package_data={'': ["data/*/*.png"]},
    packages=find_packages(exclude=['test'])
)
