from ctypes import Union
from glob import glob
import os.path as _os
import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml
from tud_rl import logger

WEIGHT_EXTENSIONS = ["actor","critic","dqn","accddqn","ensemble"]

class ConfigFile:
    """Configuration class for storing the parsed 
    yaml config file.

    Contains all configuration options possible
    for training with the `tud_rl` package.

    """
    mode: str
    seed: int
    timesteps: int
    epoch_length: int
    eval_episodes: int

    actor_weights: str
    critic_weights: str

    input_norm: bool
    input_norm_prior: str
    gamma: float
    eps_init: float
    eps_final: float
    eps_decay_steps: int
    tgt_update_freq: int
    net_struc: Sequence[Tuple[Tuple[str,int],str]]
    net_struc_actor: Sequence[Tuple[Tuple[str,int],str]]
    net_struc_critic: Sequence[Tuple[Tuple[str,int],str]]
    optimizer: str
    loss: str
    lr: float
    lr_actor: float
    lr_critic: float
    buffer_length: int
    grad_rescale: bool
    grad_clip: bool
    act_start_step: int
    upd_start_step: int
    upd_every: int
    batch_size: int
    device: str
    output_dir: str


    class Env:
        name: str
        max_episode_steps: int
        state_type: str
        wrappers: List[str]
        wrapper_kwargs: Dict[Any, Any]
        env_kwargs: Dict[Any, Any]
        info: str

    class Agent:
        pass

    def __init__(self, file: str) -> None:

        self.file = file
        self.weights = []

        # Load the YAML or JSON file into a dict
        if self.file.lower().endswith(".json"):
            self.config_dict = self._read_json(self.file)
        elif self.file.lower().endswith(".yaml"):
            self.config_dict = self._read_yaml(self.file)

        # Set the config file entries as attrs of the class
        self._set_attrs(self.config_dict)

    def _set_attrs(self, config_dict: Dict[str, Any]) -> None:

        for key, val in config_dict.items():
            if key in ["env", "agent"]:
                self._set_subclass(key, val)
            else:
                setattr(self, key, val)

    def _read_yaml(self, file_path: str) -> Dict[str, Any]:

        with open(file_path) as yaml_file:
            file = yaml_file.read()

        return yaml.safe_load(file)

    def _read_json(self, file_path: str) -> Dict[str, Any]:
        """Accept json configs for backwards compatibility"""

        logger.warning(
            "The use of `.json` configuration file is "
            "deprecated. Please define your file in the `.yaml` format."
        )

        with open(file_path) as json_file:
            file = json.load(json_file)

        for key in [
            "seed", "timesteps", "epoch_length",
            "eval_episodes", "eps_decay_steps", "tgt_update_freq",
            "buffer_length", "act_start_step",
            "upd_start_step", "upd_every", "batch_size"
        ]:
            file[key] = int(file[key])

        file["env"]["max_episode_steps"] = int(file["env"]["max_episode_steps"])

        return file

    def _set_subclass(self, key: str, d: Dict[str, Any]) -> None:

        if key == "env":
            for key_sub, val in d.items():
                setattr(self.Env, key_sub, val)

        elif key == "agent":
            for key_sub, val in d.items():
                setattr(self.Agent, key_sub, val)

    def overwrite(self, **kwargs) -> None:

        for key, val in kwargs.items():

            if hasattr(self, key):

                # overwrite config_dict for logging
                self.config_dict[key] = val

                # set attribute for easy access
                setattr(self, key, val)

            else:
                logger.error(
                    f"Overwrite: `{type(self).__name__}` has "
                    f"no attribute `{key}`. "
                    "Skipping..."
                )
                raise AttributeError

    def max_episode_handler(self) -> None:
        if self.Env.max_episode_steps == -1:
            self.Env.max_episode_steps = np.inf
            logger.info("Max episode steps set to `Inf`.")
    
    def set_weights(self, path: str) -> None:

        if not _os.isdir(path):
            msg = f"`{path}` is not a valid directory. Aborting."
            logger.error(msg)
            raise OSError(msg)
        
        # Match all files with weight extensions
        files = [glob(_os.join(path,f"*.{ext}")) for ext in WEIGHT_EXTENSIONS]
        files = [item for sublist in files for item in sublist]
        msg = " ".join([_os.basename(file) for file in files])
        logger.info(
            f"Successfully loaded weight file(s): {msg}")
        
        self.weights = files
