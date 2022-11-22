
"""
Main script for running the tud_rl package 
from inside an editor/IDE.

Basically the same as __main__.py but 
without the argument parser. 


"""
from getpass import getuser
import tud_rl.run.train_continuous as cont
import tud_rl.run.visualize_continuous as vizcont
import tud_rl.run.train_discrete as discr
import tud_rl.run.visualize_discrete as vizdiscr


from tud_rl.common.configparser import ConfigFile
from tud_rl.agents import validate_agent, is_discrete
from tud_rl.configs.continuous_actions import __path__ as cont_path
from tud_rl.configs.discrete_actions import __path__ as discr_path

# ---------------- User Settings -----------------------------
# ------------------------------------------------------------

TASK        = "train"         # ["train", "viz"]
CONFIG_FILE = "example.json"  # Your configuration file as `.yaml` or `.json`
SEED        = None            # Set a seed different to the one specified in your config
AGENT_NAME  = "RecDQN"        # Agent to train/viz with.
WEIGHTS     = None            # Path to a weight file for weight initialization

_BASEDIR = f"/home/{getuser()}/Dropbox/TU Dresden/hpc/experiments"
_BASEDIR = f"/home/{getuser()}/Dropbox/TU Dresden/hpc/aggregator/exp/DQN"
#---------------------------------------------------------------------

TASK        = "viz"
CONFIG_FILE = "pathfollower_hw.yaml"
SEED        = None
AGENT_NAME  = "DQN"
#AGENT_NAME  = "KEBootDQN"
WEIGHTS     = f"{_BASEDIR}/DQN-PathFollower-v0-downstream-06-04-20°-broadriver-2022-09-12--86880"
#WEIGHTS     = f"{_BASEDIR}/KEBootDQN-PathFollower-v0-06-04-20°-10seg-2022-11-10--34158"

# ------------------------------------------------------------
# ------------------------------------------------------------

if AGENT_NAME[-1].islower():
    validate_agent(AGENT_NAME[:-2])
    discrete = is_discrete(AGENT_NAME[:-2])
else:
    validate_agent(AGENT_NAME)
    discrete = is_discrete(AGENT_NAME)

# get the configuration file path depending on the chosen mode
base_path = discr_path[0] if discrete else cont_path[0]
config_path = f"{base_path}/{CONFIG_FILE}"

# parse the config file
config = ConfigFile(config_path)

# potentially overwrite seed
if SEED is not None:
    config.overwrite(seed=SEED)

# consider weights
if WEIGHTS is not None:
    config.set_weights(WEIGHTS)

# handle maximum episode steps
config.max_episode_handler()

if TASK == "train":
    if discrete:
        discr.train(config, AGENT_NAME)
    else:
        cont.train(config, AGENT_NAME)
elif TASK == "viz":
    if discrete:
        vizdiscr.test(config, AGENT_NAME)
    else:
        vizcont.test(config, AGENT_NAME)
