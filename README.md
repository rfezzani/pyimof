# Pyimof

A python package for optical flow estimation.

## Building

Pyimov is a pure python package. It's only dependence is
[scikit-image](https://github.com/scikit-image/scikit-image).

### `venv` users

```bash
python -m venv pyimov-dev
source pyimov-dev/bin/activate
pip install requirements.txt
pip install -e .
```

### `conda` users

```bash
conda create -y -n pyimov-dev python=3.7
conda activate pyimov-dev
conda install -y --file requirements.txt
pip install -e . --no-deps
```
