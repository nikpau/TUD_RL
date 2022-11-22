import importlib.util as _il
import inspect
import os
import sys
from typing import Sequence, Tuple, Type
from tud_rl import logger

import gym
from gym.envs.registration import EnvSpec


def spec_gen(id: str, path: os.PathLike) -> EnvSpec:
    """
    Generate a gym.EnvSpec
    which will be used to dynamically register a 
    gym Env at ``path``, with the name given by ``id``
    """

    _classes = []

    # Load the python file at `path` as 
    # a module at `tud_rl.__currentenv__`
    module_spec = _il.spec_from_file_location("tud_rl.__currentenv__",path)
    mod = _il.module_from_spec(module_spec)
    sys.modules["tud_rl.__currentenv__"] = mod
    module_spec.loader.exec_module(mod)
    for _, obj in inspect.getmembers(mod):
        if (inspect.isclass(obj) 
        and all(hasattr(obj, a) for a in ["reset","step","render"])
        and issubclass(obj,gym.Env)):
            _classes.append(obj)
    _check_results(_classes)
    _envclass = _classes[0]
    return EnvSpec(id=id, entry_point=_envclass)

def _check_results(classes: Sequence[Tuple[str,Type[gym.Env]]]) -> None:
    if len(classes) > 1:
        raise RuntimeError(
            "More than one Environment found in file at provided path or in its imports. "
            f"Found classes {classes}. "
            "Please make sure to specify only one environment per file."
        )
    elif not classes:
        raise RuntimeError(
            "No environment file found for provided path."
        )

def make_env(id: str, path: os.PathLike, **env_settings) ->gym.Env:

    logger.info(
        f"Registering environment {id} "
        f"from given path {path}"
        )
    spec_ = spec_gen(id,path)
    return gym.make(spec_, **env_settings)