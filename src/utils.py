from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Any, NamedTuple, Type

# from typing import Any, Dict, NamedTuple, Type
import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box, Dict, Discrete
from numpy.random import Generator
from torch import Tensor, nn
from torch.nn import Module


class BaseValueModule(nn.Module, ABC):
    """Base module class."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize BaseValueModule."""
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""


class ValueModule(BaseValueModule):
    """Neural Network with one hidden layer and elu activation."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32) -> None:
        """Initialize ValueModel."""
        super().__init__(input_dim, output_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor):
        """Forward pass."""
        return self.network(x)


# left out DuelingValueModule


class ValueModuleFactory:
    """Value Module Factory."""

    def __init__(self, class_obj: Type[BaseValueModule], **kwargs: Any) -> None:
        """Initialize ValueModuleFactory."""
        self.class_obj = class_obj
        self.kwargs = kwargs

    def create(self, input_dim: int, output_dim: int, **kwargs: Any) -> BaseValueModule:
        """Create a agent instance."""
        self.kwargs = {**kwargs, **self.kwargs}
        return self.class_obj(input_dim, output_dim, **self.kwargs)


class Experience(NamedTuple):
    """Represents an Experience."""

    obs: Any
    action: Any
    reward: Any
    next_obs: Any
    terminated: bool
    # truncated: bool
    info: Dict[str, Any]


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

    @abstractmethod
    def get_hparams(self) -> Dict[str, Any]:
        """Export Hyperparameters."""


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

    def get_hparams(self):
        """Return hyperparams."""
        return {"epsilon_min": self.epsilon_min, "epsilon_decay": self.epsilon_decay}


# left out RewardBasedEpsilonGreedy and LinearEpsilonDecay


class ReplayMemory:
    """Memory with sliding window that holds Experiences."""

    def __init__(self, rng: Generator, capacity: int) -> None:
        """Initialize ReplayMemory."""
        self.rng = rng
        self.memory = deque([], maxlen=capacity)

    def push(self, experience: Experience):
        """Add a experience to the memory."""
        # extract batch data
        states, actions, rewards, next_states, terminals = (
            experience.obs,
            experience.action,
            experience.reward,
            experience.next_obs,
            experience.terminated,
        )

        # Iterate over batch and save each experience individually
        for i in range(len(states)):  # assumption: batch dimension is the first dimension
            single_experience = Experience(
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                terminals[i],
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


# left out PriorityReplayMemory and HindsightReplayMemory


class ReplayMemoryFactory:
    """Replay memory factory."""

    def __init__(self, class_obj: Type[ReplayMemory], **kwargs: Any) -> None:
        """Initialize ReplayMemoryFactory."""
        self.class_obj = class_obj
        self.kwargs = kwargs

    def create(self, rng: Any, capacity: int, **options: Any) -> ReplayMemory:
        """Create a optimizer instance."""
        return self.class_obj(rng, capacity, **self.kwargs, **options)


class ModelUpdateMethod(ABC):
    """Model update base class."""

    def __init__(self, init_steps: int = 0) -> None:
        """Initialize ModelUpdateMethod."""
        super().__init__()
        self.steps = init_steps

    def step(self):
        """Execute next step for the update method."""
        self.steps += 1

    @abstractmethod
    def get_updated_state_dict(self, local: Module, target: Module) -> Dict[str, Any]:
        """Update the Model."""

    @abstractmethod
    def should_update(self) -> bool:
        """Return whether the model should be updated this step."""


# left out HardUpdate


class SoftUpdate(ModelUpdateMethod):
    """Soft update: θ_target = τ*θ_local + (1 - τ)*θ_target."""

    def __init__(self, tau: float = 0.2, init_steps: int = 0) -> None:
        """Initialize SoftUpdate."""
        super().__init__(init_steps)
        self.tau = tau

    def get_updated_state_dict(self, local: Module, target: Module) -> Dict[str, Any]:
        """Return updated state dict."""
        for target_param, policy_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data,
            )
        return target.state_dict()

    def should_update(self) -> bool:
        """Check if Update is possible."""
        return True


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


# left out one_hot_encode, is_neg_inf, is_pos_inf, scale_box_to_radian, preprocess_obs from env_util.py
