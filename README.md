# entropix
[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][build-image]][build-url]
[![MIT License][license-image]][license-url]


[release-image]:https://img.shields.io/github/release/akb89/entropix.svg?style=flat-square
[release-url]:https://github.com/akb89/entropix/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/entropix.svg?style=flat-square
[pypi-url]:https://pypi.org/project/entropix/
[build-image]:https://img.shields.io/github/workflow/status/akb89/entropix/CI?style=flat-square
[build-url]:https://github.com/akb89/entropix/actions?query=workflow%3ACI
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt

Generate count-based Distributional Semantic Models by sampling SVD singular vectors instead of using top components.

## Install
```shell
pip install entropix
```

or, after a git clone:
```shell
python3 setup.py install
```

## Use

### Sequential mode
```shell
entropix sample \
--model /abs/path/to/dense/numpy/model.npy \
--vocab /abs/path/to/corresponding/model.vocab \
--dataset dataset_to_optimize_on \  # men, simlex or simverb
--shuffle \
--mode seq \
--kfold-size .2 \  # size of kfold, between 0 and .5
--metric pearson \  # spr(spearman), pearson, rmse or both (spr+rmse)
--num-threads 5
```

### Limit mode
```shell
entropix sample \
--model /abs/path/to/dense/numpy/model.npy \
--vocab /abs/path/to/corresponding/model.vocab \
--dataset dataset_to_optimize_on \  # men, simlex or simverb
--mode limit \
--metric pearson \
--limit 10  # number of dimensions to sample
```
