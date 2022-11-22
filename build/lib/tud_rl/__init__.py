import logging
import sys
import gym
from .common.formatter import ColoredFormatter

__version__ = "1.1.0"

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = ColoredFormatter()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)