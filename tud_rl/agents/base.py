import pickle
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tud_rl import logger
from tud_rl.common.logging_func import EpochLogger
from tud_rl.common.nets import *
from tud_rl.common.buffer import *
from tud_rl.common.exploration import Gaussian_Noise,OU_Noise
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.normalizer import Input_Normalizer

Actor = Union[LSTM_Actor,GaussianActor,LSTM_GaussianActor]
Critic = Union[TQC_Critics,LSTM_Critic,LSTM_Double_Critic]
Net = Union[
    MinAtar_BootDQN,MinAtar_CoreNet,
    MinAtar_DQN,RecDQN,MLP,Double_MLP
]
Noise = Union[Gaussian_Noise,OU_Noise]
Buffer = Union[
    UniformReplayBuffer,UniformReplayBuffer_BootDQN,
    UniformReplayBuffer_LSTM,UniformReplayBufferEnvs,
    UniformReplayBufferEnvs_BootDQN
]

Optimizer = Union[torch.optim.Adam,torch.optim.RMSprop]

class _Agent(ABC):
    """Abstract Base Class for any agent
    defining its strucure.
    """
    # Name of agent
    name: str

    # Internal mode of agent ["train","test"]
    mode: str

    # Input Normalizer
    inp_normalizer: Optional[Input_Normalizer]

    # Nets for training
    DQN: Optional[Net]
    DQN_A: Optional[Net]
    DQN_B: Optional[Net]
    EnsembleDQN: Optional[Sequence[Net]]

    actor: Optional[Actor]
    critic: Optional[Critic]

    # Experience replay buffer
    buffer: Buffer

    # Optimizer
    optimizer_name: str
    optimizer: Union[Optimizer,Sequence[Optimizer]]

    # Logger
    logger: Optional[EpochLogger]

    @abstractmethod
    def select_action(self, s: np.ndarray) -> Union[int, np.ndarray]:
        """Select an action for the agent to take.
        Must take in a state and output an action.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """Trains the agent
        """
        raise NotImplementedError

    @abstractmethod
    def memorize(self, s: np.ndarray,
                 a: Union[int, np.ndarray],
                 r: float, s2: np.ndarray,
                 d: bool) -> None:
        """Add transitions tuple to the experience 
        replay buffer
        """
        raise NotImplementedError

    @abstractmethod
    def print_params(self, n_params: Union[Tuple[int, int], int], case: int) -> None:
        """Prints the number of trainable parameters of an Agent

        Args:
            n_params (int): Number of params of the net
                            If case == 0 (discrete):
                                n_params = n_params
                            If case == 1 (continuous):
                                n_params[0]: n_params actor
                                n_params[1]: n_params critic
            case (int): case [0: discrete, 1: continous]
        """
        raise NotImplementedError


class BaseAgent(_Agent):
    def __init__(self, c: ConfigFile, agent_name: str):

        # attributes and hyperparameters
        self.name             = agent_name
        self.num_actions      = c.num_actions
        self.mode             = c.mode
        self.state_shape      = c.state_shape
        self.state_type       = c.Env.state_type
        self.input_norm       = c.input_norm
        self.input_norm_prior = c.input_norm_prior
        self.gamma            = c.gamma
        self.optimizer_name   = c.optimizer
        self.loss             = c.loss
        self.buffer_length    = c.buffer_length
        self.grad_clip        = c.grad_clip
        self.grad_rescale     = c.grad_rescale
        self.act_start_step   = c.act_start_step
        self.upd_start_step   = c.upd_start_step       
        self.upd_every        = c.upd_every  # used in training files, purely for logging here
        self.batch_size       = c.batch_size
        self.device           = c.device
        self.seed             = c.seed

        # checks
        assert c.mode in ["train", "test"], "Unknown mode. Should be 'train' or 'test'."

        if self.input_norm:
            assert not (self.mode == "test" and self.input_norm_prior is None), \
                "Please supply 'input_norm_prior' in test mode with input normalization."

        assert self.state_type in ["image", "feature"],\
            "'state_type' can be either 'image' or 'feature'."

        if self.state_type == "image":
            assert len(self.state_shape) == 3 and type(self.state_shape) == tuple, \
                "'state_shape' should be: (in_channels, height, width) for images."

            if self.input_norm:
                raise NotImplementedError("Input normalization is not available for images.")

        assert self.loss in ["SmoothL1Loss", "MSELoss"], "Pick 'SmoothL1Loss' or 'MSELoss', please."
        assert self.optimizer_name_name in ["Adam", "RMSprop"], "Pick 'Adam' or 'RMSprop' as optimizer, please."
        assert self.device in ["cpu", "cuda"], "Unknown device."

        # gpu support
        if self.device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            logger.info("Using GPU support.")

        # input normalizer
        if self.input_norm:

            if self.input_norm_prior is not None:
                with open(self.input_norm_prior, "rb") as f:
                    prior = pickle.load(f)
                self.inp_normalizer = Input_Normalizer(state_dim=self.state_shape, prior=prior)
            else:
                self.inp_normalizer = Input_Normalizer(state_dim=self.state_shape, prior=None)

    def _count_params(self, net):
        """Count the number of parameters of a given net"""
        return sum([np.prod(p.shape) for p in net.parameters()])

    def print_params(self, n_params: Union[Tuple[int, int], int], case: int) -> None:
        """Prints the number of trainable parameters of an Agent

        Args:
            n_params (int): Number of params of the net
            case (int): case [0: discrete, 1: continous]
        """
        if case == 0:
            print("--------------------------------------------")
            print(f"Trainable Parameters: {n_params}")
            print("--------------------------------------------")
        else:
            print("--------------------------------------------")
            print(f"Trainable Parameters Actor: {n_params[0]}\n"
                  f"Trainable Parameters Critic: {n_params[1]}")
            print("--------------------------------------------")
