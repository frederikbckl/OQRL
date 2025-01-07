"""Simplified DQN module."""

import functools
from typing import Any, List

import numpy as np
import torch
from gymnasium import Space
from numpy.random import Generator
from torch import Tensor

from agent import Agent
from optim import SimulatedAnnealing
from utils import (
    EpsilonGreedy,
    Experience,
    ExplorationMethodFactory,
    ModelUpdateMethod,
    ReplayMemory,
    ReplayMemoryFactory,
    SoftUpdate,
    ValueModule,
    ValueModuleFactory,
)


class DQN(Agent):
    """Deep Q-Network Algorithm."""

    def __init__(
        self,
        obs_space: Space[Any],
        act_space: Space[Any],
        rng: Generator,
        device: str = "cpu",
        func_approx: ValueModuleFactory = ValueModuleFactory(ValueModule),
        memory: ReplayMemoryFactory = ReplayMemoryFactory(ReplayMemory),
        exploration_method: ExplorationMethodFactory = ExplorationMethodFactory(EpsilonGreedy),
        alpha: float = 1e-3,
        gamma: float = 0.9,
        capacity: int = 50000,
        batch_size: int = 64,
        learning_iterations: int = 1,
        learn_every: int = 1,
        target_net_update_method: ModelUpdateMethod = SoftUpdate(),
        clip_grads: bool = False,
        clip_value: float = 2.0,
    ) -> None:
        """Initialize DQN."""
        super().__init__(obs_space, act_space, rng, device)

        if capacity < batch_size:
            raise ValueError("Capacity cannot be greater than batch size.")

        if learn_every <= 0:
            raise ValueError("Steps between each learning iteration must be at least 1.")

        self.alpha = alpha
        self.gamma = gamma

        # How often/much should we learn?
        self.learn_every = learn_every
        self.learn_iterations = learning_iterations

        # Exploration method
        self.exploration_method = exploration_method.create(rng)

        # Replay Memory
        self.memory = memory.create(rng, capacity)
        self.batch_size = batch_size

        # DNNs
        self.policy_net = func_approx.create(self.obs_dim, self.act_dim).to(self.device)

        # Target net
        self.target_net = func_approx.create(self.obs_dim, self.act_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net_update_method = target_net_update_method

        # Quick optimizations
        self.clip_grads = clip_grads
        self.clip_value = clip_value

        # Optimizer and loss function
        self.optimizer = SimulatedAnnealing(
            self.policy_net.parameters(),
            init_temp=1.0,
            cooling_rate=0.999,
            min_temp=0.1,
        )  # optim.Adam(self.policy_net.parameters(), lr=self.alpha)

    def sample(self, obs: Any) -> Any:
        """Sample in DQN."""
        # Use exploration method
        if self.exploration_method.should_explore():
            return self.act_space.sample()
        # Else use policy
        return self.policy(obs)

    def policy(self, obs: Any) -> Any:
        """Define policy."""
        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.policy_net(torch.tensor(obs).to(self.device).unsqueeze(0)).squeeze()
            return np.int_(q_values.cpu().argmax().numpy())

    def update(self, exp: Experience) -> None:
        """Safe experience and update target net."""
        # Save experience in memory

        self.memory.push(exp)

        # Learn from memory
        if len(self.memory) < self.batch_size:
            return

        if self.step % self.learn_every == 0:
            for _ in range(self.learn_iterations):
                batch = self.memory.sample(self.batch_size)
                # remove infos
                batch = [entry[:-1] for entry in batch]
                experiences = [torch.tensor(np.asarray(e)).to(self.device) for e in zip(*batch)]
                self.optimizer.step(functools.partial(self.criterion, experiences))

        # Update target net
        if self.target_net_update_method.should_update():
            self.target_net.load_state_dict(
                self.target_net_update_method.get_updated_state_dict(
                    self.policy_net,
                    self.target_net,
                ),
            )

    def criterion(self, exp: List[Tensor]) -> float:
        """Update the policy network using the given experience."""
        self.policy_net.train()
        self.target_net.eval()

        states, actions, rewards, next_states, terminateds = exp  # testweise truncateds entfernt

        terminals = terminateds  # truncateds

        # old line of code
        q_values: Tensor = (
            self.policy_net(states).gather(1, actions.type(torch.int64).unsqueeze(1)).squeeze()
        )

        # Get non-terminal states and their indices

        # new add, can be removed if approach to make it run is false
        terminals = terminals.to(torch.bool)  # added

        non_terminal_mask = ~terminals
        non_terminal_next_states = next_states[non_terminal_mask]

        # Compute next state values only for non-terminal states
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_terminal_mask] = (
            self.target_net(non_terminal_next_states).max(1)[0].detach()
        )

        target_q_values = (
            rewards.float() + (1.0 - terminals.float()) * self.gamma * next_state_values
        )

        # loss = self.criterion(target_q_values, q_values)
        # self.optimizer.zero_grad()
        # loss.backward()

        # Quick fix for unstable training, avoid taking to large steps in wrong direction
        # if self.clip_grads:
        #     nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)

        # self.optimizer.step()

        td_error = target_q_values - q_values
        loss = torch.mean(td_error**2).item()

        return loss

    def on_step_end(self) -> None:
        """Increase step variable."""
        super().on_step_end()
        self.target_net_update_method.step()

    def on_episode_end(self, ep_steps: int, ep_reward: float) -> None:
        """Increase episode variable."""
        super().on_episode_end(ep_steps, ep_reward)
        self.exploration_method.step(self.step, ep_reward)


"""Simplified DQN module."""

# class DQN:
#     """Basic implementation of the Deep Q-Network algorithm."""

#     def __init__(self, obs_space, act_space, rng, device="cpu"):
#         """Initialize basic settings for DQN."""
#         self.obs_space = obs_space
#         self.act_space = act_space
#         self.rng = rng
#         self.device = device

#     def policy(self, obs):
#         """Simple policy to select an action."""
#         return self.act_space.sample()  # Simplified to just sample an action

#     def update(self, exp):
#         """Placeholder for the update method."""
#         pass  # No operation for now

#     def on_step_end(self) -> None:
#         """Increase step variable. Called at end of each step (after agent interacts with env)."""
#         super().on_step_end()
#         self.target_net_update_method.step()

#     def on_episode_end(self, ep_steps: int, ep_reward: float) -> None:
#          """Increase episode variable. Called at end of each episode (when done flag becomes true)."""
#         super().on_episode_end(ep_steps, ep_reward)
#         self.exploration_method.step(self.step, ep_reward)
