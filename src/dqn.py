"""Simplified DQN module."""

import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Module

from agent import Agent
from experience import Experience
from optim import SimulatedAnnealing
from utils import EpsilonGreedy, ExplorationMethodFactory, ReplayMemory, ReplayMemoryFactory

# add ModelUpdateMethod
# add ValueModuleFactory


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


class HardUpdate(ModelUpdateMethod):
    """Overwrites target model in a fixed interval."""

    def __init__(self, target_update_interval: int, init_steps: int = 0) -> None:
        """Initialize HardUpdate."""
        super().__init__(init_steps)
        self.target_update_interval = target_update_interval

    def get_updated_state_dict(self, local: Module, target: Module) -> Dict[str, Any]:
        """Get updated state dict."""
        return local.state_dict()

    def should_update(self) -> bool:
        """Update."""
        return self.steps % self.target_update_interval == 0


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


class DQN(Agent):
    """Basic implementation of the Deep Q-Network algorithm."""

    def __init__(
        self,
        obs_space,
        act_space,
        rng,
        device="cpu",
        func_approx: ValueModuleFactory = ValueModuleFactory(ValueModule),
        memory: ReplayMemoryFactory = ReplayMemoryFactory(ReplayMemory),
        exploration_method: ExplorationMethodFactory = ExplorationMethodFactory(EpsilonGreedy),
        capacity: int = 50000,
        batch_size: int = 64,
        learning_iterations: int = 1,
        learn_every: int = 1,
        target_net_update_method: ModelUpdateMethod = SoftUpdate(),
    ):
        """Initialize basic settings for DQN."""
        super().__init__(obs_space, act_space, rng, device)
        # self.obs_space = obs_space
        # self.act_space = act_space
        # self.rng = rng
        # self.device = device
        # self.step = 0
        # self.episode = 0

        if capacity < batch_size:
            raise ValueError("Capacity cannot be greater than batch size.")

        if learn_every <= 0:
            raise ValueError("Steps between each learning iteration must be at least 1.")

        # add alpha / gamma if needed

        # How often to learn
        self.learn_every = learn_every
        self.learn_iterations = learning_iterations

        # Exploration method
        self.exploration_method = exploration_method.create(rng)

        # Replay Memory
        self.memory = memory.create(rng, capacity)
        self.batch_size = batch_size

        # DNN's
        self.policy_net = func_approx.create(self.obs_dim, self.act_dim).to(self.device)

        # Target net
        self.target_net = func_approx.create(self.obs_dim, self.act_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net_update_method = target_net_update_method

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
        """Save experience and update target net."""
        # Save experience in memory

        # self.memory.push(exp)

        # Learn from memory
        if len(self.memory) < self.batch_size:
            return  # early exit if replay memory does not have batch_size

        if self.step % self.learn_every == 0:  # updates policy network every self.learn_every
            for _ in range(self.learn_iterations):  # repeats for self.learn_iterations steps
                batch = self.memory.sample(
                    self.batch_size,
                )  # sample batch of experiences from ReplayMemory
                batch = [
                    entry[:-1] for entry in batch
                ]  # remove additional info (not needed for training)
                experiences = [
                    torch.tensor(np.asarray(e)).to(self.device) for e in zip(*batch)
                ]  # converts batch into PyTorch tensors and transfers to device
                self.optimizer.step(
                    functools.partial(self.criterion, experiences),
                )  # loss function (criterion) is applied to sampled experiences + optimizer adjusts weights

        # Update target net
        if self.target_net_update_method.should_update():
            self.target_net.load_state_dict(
                self.target_net_update_method.get_updated_state_dict(
                    self.policy_net,
                    self.target_net,
                ),
            )

    def on_step_end(self) -> None:
        """Increase step variable."""
        super().on_step_end()
        self.target_net_update_method.step()

    def on_episode_end(self, ep_steps: int, ep_reward: float) -> None:
        """Increase episode variable."""
        super().on_episode_end(ep_steps, ep_reward)
        self.exploration_method.step(self.step, ep_reward)
