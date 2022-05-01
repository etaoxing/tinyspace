<!-- start about -->

[pypi-url]: https://pypi.python.org/pypi/tinyspace
[license-badge]: https://img.shields.io/pypi/l/tinyspace.svg
[version-badge]: https://img.shields.io/pypi/v/tinyspace.svg
[pyversion-badge]: https://img.shields.io/pypi/pyversions/tinyspace.svg

[tests-badge]: https://github.com/etaoxing/tinyspace/actions/workflows/tests.yml/badge.svg
[tests-url]: https://github.com/etaoxing/tinyspace/actions/workflows/tests.yml

[docs-badge]: https://img.shields.io/readthedocs/tinyspace.svg
[docs-url]: https://tinyspace.readthedocs.io/

# 🤏 tinyspace

[![license][license-badge]][pypi-url]
[![version][version-badge]][pypi-url]
[![pyversion][pyversion-badge]][pypi-url]
[![tests][tests-badge]][tests-url]
[![docs][docs-badge]][docs-url]

A simple and lightweight spaces implementation for RL environments, in place of `gym.spaces`.
<!-- end about -->


<!-- start quickstart -->
# Quickstart

```bash
pip install tinyspace
```
<!-- end quickstart -->

<!-- start example -->
# Example

```python
from tinyspace import TinySpace, Space

action_space = TinySpace(shape=(), dtype=np.int, low=0, high=10, desc="action space", cls="discrete")
if action_space["cls"] == "discrete":  # access like a dictionary
    ...
elif action_space.cls == "box":  # or dot access
    ...

observation_space = TinySpace(shape=(3, 224, 224), dtype=torch.uint8, low=0, high=255)  # a valid `Space`
_nd_shape = (-1, 3)  # can use `-1` or `None` for variable-length dimensions
_pcd_space = TinySpace(shape=_nd_space, dtype=np.float32, low=-np.inf, high=np.inf, desc="partial point cloud")
observation_space = dict(  # dict where each value is a `TinySpace` is also a valid `Space`
    rgb=observation_space,
    endeffector_pos=TinySpace(shape=(3,), dtype=np.float32, low=-np.inf, high=np.inf),
    pcd=_pcd_space,
)

def check_obs(obs, space: Space):  # use `Space` type for either `TinySpace` or dict of `TinySpace`
    if isinstance(space, TinySpace):
        low = space["low"]  # preferred, so that space can also just be a standard dict
        high = space.high  # but could also use dot access if you don't need that use case
        ...
    else:
        return {k: check_obs(obs[k], v) for k, v in space.items()}
```
<!-- end example -->
