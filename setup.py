from setuptools import setup, find_packages


descr = """Pyimof

Implementation of the popular TV-L1 algorithm and the robust iLK
method for optical flow estimation. Pyimof is provided battery
included: test data, IO and visualization tools are provided.

Please refer to the online documentation at
https://pyimof.rtfd.io/en/latest/
"""

DISTNAME = 'pyimof'
VERSION = '1.0.0'
DESCRIPTION = 'Python package for optical flow estimation'
LONG_DESCRIPTION = descr
MAINTAINER = 'Riadh Fezzani'
MAINTAINER_EMAIL = 'rfezzani@gmail.com'
URL = 'https://pyimof.rtfd.io/en/latest/'
LICENSE = 'GPL-3.0'
DOWNLOAD_URL = 'https://github.com/rfezzani/pyimof/archive/master.zip'
PROJECT_URLS = {
    "Bug Tracker": 'https://github.com/rfezzani/pyimof/issues',
    "Documentation": 'https://pyimof.rtfd.io/en/latest/',
    "Source Code": 'https://github.com/rfezzani/pyimof'
}


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if l]

    return requires


REQUIRES = parse_requirements_file('requirements.txt')
EXTRA_REQUIRES = {'docs': parse_requirements_file('doc/requirements.txt')}

setup(
    name=DISTNAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    license=LICENSE,
    download_url=DOWNLOAD_URL,
    project_urls=PROJECT_URLS,
    version=VERSION,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    install_requires=REQUIRES,
    extras_require=EXTRA_REQUIRES,
    python_requires='>=3.5',
    zip_safe=False,
    package_data={'': ["data/*/*.png"]},
    packages=find_packages(exclude=['test', 'doc'])
)
