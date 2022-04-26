import dataclasses
from typing import Dict, List, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np

Arr = np.ndarray
Dtype = np.dtype

try:
    import torch

    Arr = Union[Arr, torch.Tensor]
    Dtype = Union[Dtype, torch.dtype]

    # see https://discuss.pytorch.org/t/converting-a-numpy-dtype-to-torch-dtype/52279

    # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
    numpy_to_torch_dtype_dict = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }

    # numpy dtypes like np.float64 are not instances, but rather classes. This leads to rather absurd cases like
    # np.float64 != np.dtype("float64") but np.float64 == np.dtype("float64").type.
    # Especially when checking against a reference we can't be sure which variant we get, so we simply try both.
    def numpy_to_torch_dtype(np_dtype):
        try:
            return numpy_to_torch_dtype_dict[np_dtype]
        except KeyError:
            return numpy_to_torch_dtype_dict[np_dtype.type]

except ImportError:
    pass


@dataclasses.dataclass
class TinySpace:
    r"""
    A simple and lightweight spaces implementation for RL environments, in place of `gym.spaces`.
    Any `Space` is a `TinySpace` with the following args or a `dict` where each value is a `TinySpace`.

    You can either use this dataclass or just define a `Space` using dicts, since tinyspace
    prefers subscriptable access. However, because a TinySpace is a dataclass, dot access also works.

    Example:
        ```python
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

    Args:
        shape: Shape of space. Use `-1` or `None` for variable-length dimensions.
        dtype: Data type of space.
        low: Lower bound of space.
        high: Upper bound of space.
        desc: Description of space.
        cls: Type of space, such as ["discrete", "box"].
    """
    shape: Tuple
    dtype: Dtype
    low: Union[int, float, Arr]
    high: Union[int, float, Arr]
    cls: Optional[str] = None
    desc: Optional[str] = None

    def get(self, key, default=None):
        value = getattr(self, key, default)
        if value is None:  # also set to default if space.key exists but is set to None
            value = default
        return value

    def __getitem__(self, item):  # makes subscriptable
        return getattr(self, item)


Space = Union[TinySpace, Mapping[str, TinySpace]]
T = TypeVar("T")


def space_from_dict(d: Union[Dict[str, T], Dict[str, Dict]]):
    if "shape" in d.keys():
        return TinySpace(**d)
    else:
        return {k: space_from_dict(v) for k, v in d.items()}


def from_gym_space(gym_space, to_tinyspace=True) -> Space:
    import gym.spaces

    if isinstance(gym_space, gym.spaces.Dict):
        out_space = {k: from_gym_space(v, to_tinyspace=to_tinyspace) for k, v in gym_space.items()}
    else:
        out_space = dict(
            shape=gym_space.shape,
            dtype=gym_space.dtype,
        )
        if isinstance(gym_space, gym.spaces.Box):
            low = gym_space.low
            low_unique = np.unique(low)
            if len(low_unique) == 1:
                low = low_unique[0]

            high = gym_space.high
            high_unique = np.unique(high)
            if len(high_unique) == 1:
                high = high_unique[0]

            out_space.update(
                cls="box",
                low=low,
                high=high,
            )
        elif isinstance(gym_space, gym.spaces.Discrete):
            out_space.update(
                cls="discrete",
                low=0,
                high=gym_space.n,
            )
        elif isinstance(gym_space, gym.spaces.MultiDiscrete):
            out_space.update(
                cls="multidiscrete",
                low=np.zeros_like(gym_space.nvec),
                high=gym_space.nvec,
            )
        else:
            raise NotImplementedError

        if to_tinyspace:
            out_space = TinySpace(**out_space)

    return out_space


def to_gym_space(space: Space):
    import gym.spaces

    if isinstance(space, TinySpace) or "shape" in space.keys():
        _cls = space.get("cls", "box")
        if _cls == "discrete":
            s = gym.spaces.Discrete(space["high"])
        elif _cls == "multidiscrete":
            s = gym.spaces.MultiDiscrete(space["high"])
        elif _cls == "box":
            s = gym.spaces.Box(
                low=space["low"],
                high=space["high"],
                shape=space["shape"],
                dtype=space["dtype"],
            )
        else:
            raise NotImplementedError
        return s
    else:
        s = {k: to_gym_space(v) for k, v in space.items()}
        s = gym.spaces.Dict(s)
        return s


def convert_gymenv_spaces(env, to_tinyspace=True):
    env.action_space = from_gym_space(env.action_space, to_tinyspace=to_tinyspace)
    env.observation_space = from_gym_space(env.observation_space, to_tinyspace=to_tinyspace)
    return env


def _is_nd(d) -> bool:
    return d == -1 or d is None


def is_ndshape(space: Space) -> bool:
    if isinstance(space, TinySpace) or "shape" in space.keys():
        return any([_is_nd(d) for d in space["shape"]])
    else:
        o = {}
        for k, v in space.items():
            o[k] = is_ndshape(v)
        return o


def sample_from_space(
    space: Space,
    zeros: bool = True,
    batch_size: int = 1,
    ndim_pad: int = 1,
    to_torch_tensor: bool = False,
) -> Union[Dict, Arr]:
    if isinstance(space, TinySpace) or "shape" in space.keys():
        if batch_size:
            if not (isinstance(batch_size, list) or isinstance(batch_size, tuple)):
                batch_size = (batch_size,)
            size = (*batch_size, *space["shape"])
        size = tuple(ndim_pad if _is_nd(d) else d for d in size)
        dtype = space["dtype"]
        if type(dtype).__module__ == "torch" or to_torch_tensor:
            if type(dtype).__module__ != "torch":
                dtype = numpy_to_torch_dtype(dtype)
            return torch.zeros(size, dtype=dtype)
        else:
            return np.zeros(size, dtype=dtype)
    else:
        o = {}
        for k, v in space.items():
            o[k] = sample_from_space(
                v,
                zeros=zeros,
                batch_size=batch_size,
                ndim_pad=ndim_pad,
                to_torch_tensor=to_torch_tensor,
            )
        return o


def collate_obs(obses: Union[List, Tuple], space: Space, to_torch_tensor: bool = False):
    r"""
    Args:
        obses: Batch of observations which need to be collated.
        space: A `TinySpace` or `dict` where each value is a `TinySpace`.
        to_torch_tensor: If True, then convert to `torch.Tensor`.
    """
    assert len(obses) > 0, "need observations from at least one environment"
    assert isinstance(obses, (list, tuple)), "expected list or tuple of observations per environment"

    if isinstance(space, TinySpace) or "shape" in space.keys():
        if to_torch_tensor:
            return torch.tensor(obses)
        else:
            raise NotImplementedError
    else:
        out = {}
        for k, v in space.items():
            if to_torch_tensor:
                if is_ndshape(v):
                    out[k] = [torch.tensor(o[k]) for o in obses]
                else:
                    out[k] = torch.tensor([o[k] for o in obses])
            else:
                raise NotImplementedError
        return out
