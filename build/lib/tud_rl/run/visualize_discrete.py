import argparse
import json
import random
import gym
import numpy as np
import torch

import tud_rl.agents.discrete as agents

from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.wrappers import get_wrapper
from tud_rl.configs.discrete_actions import __path__ as c_path
from tud_rl.envs import make_env


def visualize_policy(env: gym.Env, agent: _Agent, c: ConfigFile):

    for _ in range(c.eval_episodes):

        # get initial state
        s = env.reset()

        # potentially normalize it
        if c.input_norm:
            s = agent.inp_normalizer.normalize(s, mode=agent.mode)

        cur_ret = 0
        d = False
        eval_epi_steps = 0

        while not d:

            eval_epi_steps += 1

            # render env
            #env.render()
            print(f"Timestep: {eval_epi_steps}/{c.Env.max_episode_steps}",end="\r")

            # select action
            a = agent.select_action(s)

            # perform step
            the_end = False
            try:
                s2, r, d, _ = env.step(a)
            except:
                the_end = True

            # potentially normalize s2
            if c.input_norm:
                s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c.Env.max_episode_steps or the_end or d:
                SAVE = True
                if SAVE:
                    print("Saving...")
                    path = "trajectory_plots/"
                    path = "/home/niklaspaulig/Dropbox/TU Dresden/writing/Autonomous Navigation on Rivers using Deep Reinforcement Learning/data/lorelei/dqn/"
                    with open(path + "cte", "w") as file:
                        file.write(json.dumps(list(env.history.cte)))

                    with open(path + "heading_error", "w") as file:
                        file.write(json.dumps(list(env.history.heading_error)))

                    with open(path + "rudder_movement", "w") as file:
                        file.write(json.dumps(list(env.history.delta)))

                    with open(path + "cross_curr", "w") as file:
                        file.write(json.dumps(list(env.history.cross_curr_angle)))

                    with open(path + "yaw_accel", "w") as file:
                        file.write(json.dumps(list(env.history.r)))

                    with open(path + "reward", "w") as file:
                        file.write(json.dumps(list(env.history.reward)))

                    with open(path + "pos", "w") as file:
                        file.write(json.dumps([[pos.x,pos.y] for pos in env.history.pos]))
                    
                    with open(path + "path", "w") as file:
                        file.write(json.dumps(list(env.history.path)))
                break

        print(cur_ret)


def test(c: ConfigFile, agent_name: str):
    # init env
    env = make_env(c.Env.name,c.Env.path, **c.Env.env_kwargs)

    # wrappers
    for wrapper in c.Env.wrappers:
        wrapper_kwargs = c.Env.wrappers[wrapper]
        env: gym.Env = get_wrapper(name=wrapper, env=env, **wrapper_kwargs)

    # get state_shape
    if c.Env.state_type == "image":
        assert "MinAtar" in c.Env.name, "Only MinAtar-interface available for images."

        # careful, MinAtar constructs state as
        # (height, width, in_channels), which is NOT aligned with PyTorch
        c.state_shape = (env.observation_space.shape[2],
                         *env.observation_space.shape[0:2])

    elif c.Env.state_type == "feature":
        c.state_shape = env.observation_space.shape[0]

    # mode and num actions
    c.mode = "test"
    c.num_actions = env.action_space.n

    # seeding
    env.seed(c.seed)
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    random.seed(c.seed)

    if agent_name[-1].islower():
        agent_name_red = agent_name[:-2] + "Agent"
    else:
        agent_name_red = agent_name + "Agent"

    # init agent
    agent_ = getattr(agents, agent_name_red)  # Get agent class by name
    agent: _Agent = agent_(c, agent_name)  # Instantiate agent

    # visualization
    visualize_policy(env=env, agent=agent, c=c)
