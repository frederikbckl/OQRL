from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Union

import torch
from torch import Tensor


class Optimizer(ABC):
    """Optimizer base class."""

    def __init__(self, params: Union[Iterable[Tensor], Iterable[Dict[str, Any]]]):
        self.params = params

    def get_param_tensors(self):
        """Helper method to get tensors from the params."""
        for param_group in self.params:
            if isinstance(param_group, dict):
                for _, param in param_group.items():
                    if isinstance(param, Tensor):
                        yield param
            else:
                yield param_group

    def clone_param_tensors(self) -> List[Tensor]:
        """Clones and returns a list of parameter tensors.

        Returns:
            A list of cloned parameter tensors.

        """
        return [p.clone() for p in self.get_param_tensors()]

    @abstractmethod
    def step(self, eval_func: Callable[[], float]):
        """Perform a single optimization step."""


class SimulatedAnnealing(Optimizer):
    """Simulated Annealing Optimizer."""

    def __init__(
        self,
        params: Union[Iterable[Tensor], Iterable[Dict[str, Any]]],
        init_temp: float,
        cooling_rate: float,
        min_temp: float,
    ) -> None:
        """Initialize SimulatedAnnealing."""
        super().__init__(params)
        self.temperature = init_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.state = self.clone_param_tensors()

    def perturb(self) -> None:
        """Perturbs the current parameters."""
        with torch.no_grad():
            for param in self.get_param_tensors():
                # Create a random perturbation based on the current temperature
                perturbation = torch.randn_like(param) * self.temperature
                # Update the parameters in place
                param.add_(perturbation)

    def step(self, eval_func: Callable[[], float]) -> None:
        """Perform a single optimization step."""
        # Save the current parameters before perturbing
        current_params = self.clone_param_tensors()

        # Evaluate loss with current (before perturbation) parameters
        current_loss = eval_func()

        # Perturb the parameters
        self.perturb()

        # Evaluate the new parameters with the environment
        new_loss = eval_func()

        if new_loss < current_loss:
            self.state = self.clone_param_tensors()  # Save the new best state
        else:
            # Accept the worse solution with a certain probability
            acceptance_probability = torch.exp(
                torch.tensor(current_loss - new_loss) / self.temperature,
            )
            if torch.rand(1).item() > acceptance_probability:
                # Revert to the previous parameters if not accepted
                for param, cp in zip(self.get_param_tensors(), current_params):
                    param.copy_(cp)

        # Cool down the temperature
        self.temperature = max(self.min_temp, self.temperature * self.cooling_rate)
