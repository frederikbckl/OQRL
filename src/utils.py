from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Any, Dict, Type

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box, Dict, Discrete
from numpy.random import Generator

from experience import Experience


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


class ReplayMemory:
    """Memory with silding window that holds Experiences."""

    def __init__(self, rng: Generator, capacity: int) -> None:
        """Initialize ReplayMemory."""
        self.rng = rng
        self.memory = deque([], maxlen=capacity)

    def push(self, experience: Experience):
        """Add a experience to the memory."""
        # Extract batch data
        states, actions, rewards, next_states, terminals = (
            experience.obs,
            experience.action,
            experience.reward,
            experience.next_obs,
            experience.terminated,
        )

        print("Terminals type:", type(terminals))
        print("Terminals content:", terminals)

        # Iterate over batch and save each experience individually
        for i in range(len(states)):  # Assumption: batch dimension is the first dimension
            print(f"Rewards shape: {rewards.shape}, type: {type(rewards)}")
            single_experience = Experience(
                states[i],
                actions[i],
                rewards[i],
                # rewards[i].item(),
                # rewards[i].item()
                # if isinstance(rewards[i], torch.Tensor)
                # else rewards[i],  # Convert tensor to scalar
                next_states[i],
                terminals[i],
                # bool(terminals[i])
                # if isinstance(terminals[i], (torch.Tensor, np.generic))
                # else terminals[i],
                {},
            )
            self.memory.append(single_experience)

    def sample(self, batch_size: int):
        """Sample a batchs of experiences from the memory."""
        return self.rng.choice(np.asarray(self.memory, dtype=object), batch_size)

    def get_capacity(self):
        """Get memory capacity."""
        return self.memory.maxlen

    def __len__(self):
        """Get memory length."""
        return len(self.memory)

    # def __getitem__(self, idx):
    #     """Return a single sample of data."""
    #     # print("Hello __getitem__")
    #     state = self.states[idx]
    #     action = self.actions[idx]
    #     reward = self.rewards[idx]
    #     next_observation = self.next_observations[idx]
    #     terminal = self.terminals[idx]
    #     return (
    #         torch.tensor(observation, dtype=torch.float32),
    #         torch.tensor(
    #             action,
    #             dtype=torch.long,
    #         ),  # actions are discrete in CartPole -> PyTorch requires the tensors to be torch.long
    #         torch.tensor(reward, dtype=torch.float32),
    #         torch.tensor(next_observation, dtype=torch.float32),
    #         bool(terminal),
    #     )


class ReplayMemoryFactory:
    """Replay memory factory."""

    def __init__(self, class_obj: Type[ReplayMemory], **kwargs: Any) -> None:
        """Initialize ReplayMemoryFactory."""
        self.class_obj = class_obj
        self.kwargs = kwargs

    def create(self, rng: Any, capacity: int, **options: Any) -> ReplayMemory:
        """Create a optimizer instance."""
        return self.class_obj(rng, capacity, **self.kwargs, **options)
