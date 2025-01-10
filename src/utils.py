from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Type

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box, Dict, Discrete
from numpy.random import Generator


class ActionType(Enum):
    """Action type of an environment."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


def get_obs_dim(space: Space[Any]) -> int:
    """Get observation dimension."""
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, Box):
        # return space.shape[0]
        return int(np.prod(space.shape))
    if isinstance(space, Dict):
        return sum(get_obs_dim(subspace) for subspace in space.spaces.values())
    raise NotImplementedError


def get_act_dim(space: Space[Any]) -> int:
    """Get action dimension."""
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, Box):
        return space.shape[0]
    raise NotImplementedError


def get_act_type(space: Space[Any]) -> ActionType:
    """Determine the action type of env."""
    if isinstance(space, Discrete):
        return ActionType.DISCRETE
    return ActionType.CONTINUOUS


class ExplorationMethod(ABC):
    """Exploration method base class."""

    def __init__(self, rng: Generator) -> None:
        """Initialize ExplorationMethod."""
        super().__init__()
        self.rng = rng

    @abstractmethod
    def should_explore(self) -> bool:
        """Return whether the agent should explore this step."""

    @abstractmethod
    def step(self, total_steps: int, total_reward: float):
        """Execute the next step."""

    # @abstractmethod
    # def get_hparams(self) -> Dict[str, Any]:
    #     """Export Hyperparameters."""


class EpsilonGreedy(ExplorationMethod):
    """Explores with probability epsilon."""

    def __init__(
        self,
        rng: Generator,
        epsilon: float = 1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.95,
    ) -> None:
        """Initialize EpsilonGreedy."""
        super().__init__(rng)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_key = "Epsilon"

    def should_explore(self) -> bool:
        """Explore."""
        return self.rng.random() <= self.epsilon

    def sample(self, action_space: Any) -> Any:
        """Sample."""
        return self.rng.choice(np.arange(action_space))

    def step(self, total_steps: int, total_reward: float):
        """Take a step."""
        self.epsilon = np.max([self.epsilon * self.epsilon_decay, self.epsilon_min])

    # def get_hparams(self):
    #     """Return hyperparams."""
    #     return {"epsilon_min": self.epsilon_min, "epsilon_decay": self.epsilon_decay}


class ExplorationMethodFactory:
    """Exploration method factory."""

    def __init__(self, class_obj: Type[ExplorationMethod], *args: Any, **kwargs: Any) -> None:
        """Initialize ExplorationMethodFactory."""
        self.class_obj = class_obj
        self.args = args
        self.kwargs = kwargs

    def create(self, rng: Generator):
        """Create a loss instance."""
        return self.class_obj(rng, *self.args, **self.kwargs)
