# RandomGen

A Python3 package for generating random variates.

Author: Christian D. Powell

Email: cpowell74@gatech.edu


## Installation

Installation of the `randomgen` package can be completed by cloning the git repository and installing the local package via pip.

```bash
git clone ...
cd randomgen
python3 -m pip install -e .
```

## Contents

```bash
RandomGen/
|
|-randomgen/  # Python3 package installed
| |
| |-__init__.py
| |-monte_carlo_rv.py  # module containing monte_carlo_rv object with all methods used to generated random variable distributions
|
|-scripts/
| |
| |-data_figures.ipynb  # iPython notebook used to generate all figures and tables in report/can also be used for examples of package function
| |-demo.ipynb  # legacy notebook the package was initial developed out of
| |-my_table.docx  # word formatted quantile tables
|
|-.gitignore
|-LICENSE  # MIT license
|-pyproject.toml  # toml file with metadata/requirements for package
|-README.md  # this file
```


## Quickstart

Below are examples of how to generate distributions using either the Inverse Transform Method or the Test Statistic Sampling Method. For additional examples see [this](scripts/data_figures.ipynb) iPython notebook.

### Inverse Transform Method

```python
>>> from scipy import stats 
>>> norm_dist = monte_carlo_rv(stats.norm, dist_name=”Normal”, iterations=10_000) 
>>> norm_dist.run()
100
200
...
>>> norm_dist.plot()
...
```


### Test Statistic Sampling Method

```python
>>> kolm_dist = monte_carlo_rv(“Kolmogorov”, dist_name=”Kolmogorov–Smirnov”, iterations=10_000, samples=1_000) 
>>> kolm_dist.run() 
100
200
... 
```


## License

Copyright (c) Christian D. Powell. All rights reserved.

Licensed under the [MIT](LICENSE.txt) license.
