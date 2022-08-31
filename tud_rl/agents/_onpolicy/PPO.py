from typing import Tuple, Union
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical

from ... import logger
from tud_rl.common.nets import PPOActor, PPOCritic
from tud_rl.agents.base import BaseAgent
from tud_rl.common.configparser import ConfigFile

class PPOAgent(BaseAgent):

    def __init__(self, c: ConfigFile, agent_name: str):
        super().__init__(c, agent_name)

        self.lr_actor         = c.lr_actor
        self.lr_critic        = c.lr_critic
        self.continuous       = c.ppo_continuous
        self.weights          = c.weights
        self.action_std_init  = 0.6

        if self.continuous:
            logger.info("Continuous PPO mode selected.")

        # Current policy
        self.actor = PPOActor(self.state_shape,self.num_actions,continuous=self.continuous)
        self.critic = PPOCritic(self.state_shape)

        # Old policy for comparison
        self.actor_old = PPOActor(self.state_shape,self.num_actions,continuous=self.continuous)
        self.critic_old = PPOCritic(self.state_shape)

        # number of parameters for actor and critic
        self.n_params = self._count_params(self.actor), self._count_params(self.critic)

        if self.weights is not None:
            actor_weights, critic_weights = self.weights
            self.actor.load_state_dict(torch.load(actor_weights,map_location=self.device))            
            self.critic.load_state_dict(torch.load(critic_weights,map_location=self.device))

        # Optimizer
        if self.optimizer_name == "Adam":
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        else:
            self.actor_optimizer = optim.RMSprop(
                self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            self.critic_optimizer = optim.RMSprop(
                self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)

        
        self.rollout_buffer = RolloutBuffer()

        if self.continuous:
            self.action_var = torch .full((self.num_actions,), self.action_std_init**2).to(self.device)

    def act(self, state: torch.Tensor, actor: PPOActor) -> Tuple[float,float]:

        if self.continuous:
            action_mean = actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = actor(state)
            dist = Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state: torch.Tensor, action, actor: PPOActor, critic: PPOCritic):

        if self.continuous:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.num_actions == 1:
                action = action.reshape(-1, self.num_actions)
        else:
            action_probs = actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def select_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        
        if self.continuous:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.act(state,self.actor_old)
            
            self.rollout_buffer.states.append(state)
            self.rollout_buffer.actions.append(action)
            self.rollout_buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.act(state,self.actor_old)
            
            self.rollout_buffer.states.append(state)
            self.rollout_buffer.actions.append(action)
            self.rollout_buffer.logprobs.append(action_logprob)

            return action.item()


class RolloutBuffer:
    """Rollout Buffer for MonteCarlo Playouts"""
    def __init__(self) -> None:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminal = []

    def reset(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminal.clear()