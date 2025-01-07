from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import torch
from gymnasium import Space
from numpy.random import Generator

from utils import Experience, get_act_dim, get_act_type, get_obs_dim


class Agent(ABC):
    """Agent base class."""

    def __init__(
        self,
        obs_space: Space[Any],
        act_space: Space[Any],
        rng: Generator,
        device: str = "cpu",
    ) -> None:
        """Initialize Agent."""
        self.obs_space = obs_space
        self.act_space = act_space
        self.rng = rng

        self.obs_dim = get_obs_dim(self.obs_space)
        self.act_dim = get_act_dim(self.act_space)
        self.act_type = get_act_type(self.act_space)
        self.device = torch.device(device)

        self.step = 0
        self.episode = 0

    @abstractmethod
    def policy(self, obs: Any) -> Any:
        """Return an action according to the current policy."""

    @abstractmethod
    def sample(self, obs: Any) -> Any:
        """Return an action according to the current policy with exploration."""

    @abstractmethod
    def update(self, exp: Experience) -> None:
        """Integrates the provided experience into the agent."""

    def on_step_end(self) -> None:
        """Call at the end of a step."""
        self.step += 1

    def on_episode_end(self) -> None:  # ep_steps: int and ep_reward: float got removed
        """Call at the end of an episode."""
        self.episode += 1

    def get_hyperparams(self) -> Dict[str, Any]:
        """Return the hyperparameters of the agent."""
        return {}


class AgentFactory:
    """Agent factory class."""

    def __init__(self, class_obj: Type[Agent], **kwargs: Any) -> None:
        """Initialize AgentFactory."""
        self.class_obj = class_obj
        self.kwargs = kwargs

    def create(
        self,
        obs_space: Space[Any],
        act_space: Space[Any],
        rng: Generator,
    ) -> Agent:
        """Create a agent instance."""
        return self.class_obj(obs_space, act_space, rng, **self.kwargs)

    def get_agent_name(self) -> str:
        """Return the agents class name."""
        return self.class_obj.__name__

    def get_kwargs(self) -> dict[str, Any]:
        """Return the specified kwargs."""
        return self.kwargs


# """Simplified agent module."""


# class Agent:
#     """Basic agent class."""

#     def __init__(self, obs_space, act_space, rng, device="cpu"):
#         """Initialize the agent with minimal settings."""
#         self.obs_space = obs_space
#         self.act_space = act_space
#         self.rng = rng
#         self.device = device

#     def policy(self, obs):
#         """Placeholder for the policy to return an action."""
#         return self.act_space.sample()  # Simplified to just sample an action

#     def update(self, exp):
#         """Placeholder for update method."""
#         pass  # No operation for now


# class AgentFactory:
#     """Factory class to create agent instances."""

#     def __init__(self, agent_class, **kwargs):
#         """Initialize the factory with an agent class."""
#         self.agent_class = agent_class
#         self.kwargs = kwargs

#     def create(self, obs_space, act_space, rng):
#         """Create an agent instance."""
#         return self.agent_class(obs_space, act_space, rng, **self.kwargs)
