# GabrielGraph

[![License MIT](https://img.shields.io/pypi/l/GabrielGraph.svg?color=green)](https://github.com/GuignardLab/GabrielGraph/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/GabrielGraph.svg?color=green)](https://pypi.org/project/GabrielGraph)
[![Python Version](https://img.shields.io/pypi/pyversions/GabrielGraph.svg?color=green)](https://python.org)
[![tests](https://github.com/GuignardLab/GabrielGraph/workflows/tests/badge.svg)](https://github.com/GuignardLab/GabrielGraph/actions)
[![codecov](https://codecov.io/gh/GuignardLab/GabrielGraph/branch/main/graph/badge.svg)](https://codecov.io/gh/GuignardLab/GabrielGraph)

A small class to create Gabriel Graphs

----------------------------------

## Installation

You can install `GabrielGraph` via [pip]:

```shell
pip install GabrielGraph
```

To install latest development version :

```shell
pip install git+https://github.com/GuignardLab/GabrielGraph.git
```

## Tyical use

```python
from gabrielgraph import build_gabriel_graph
import numpy as np

np.random.seed(0)
point_ids = np.arange(10)
point_positions = np.random.random((10, 3))

gg = build_gabriel_graph(point_ids, point_positions)

gg[0] # {1, 2, 3, 6, 7, 9}

gg = build_gabriel_graph(point_ids, point_positions, "adj-mat") # coo sparse array
gg = gg.todok()  # to convert to dictionary of keys sparse array
gg[0, 1] # True
gg[0, 4] # False
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"GabrielGraph" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

----------------------------------

This library was generated using [Cookiecutter] and a custom made template based on [@napari]'s [cookiecutter-napari-plugin] template.

[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[pip]: https://pypi.org/project/pip/
[tox]: https://tox.readthedocs.io/en/latest/

[file an issue]: https://github.com/GuignardLab/GabrielGraph/issues
