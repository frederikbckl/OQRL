from typing import Any, Callable, Dict, Iterable, Union

import torch


class SimulatedAnnealing:
    """Simulated Annealing Optimizer."""

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        init_temp: float,
        cooling_rate: float,
        min_temp: float,
    ):
        """Initialize the SimulatedAnnealing optimizer.

        Args:
            params: Parameters to optimize.
            init_temp: Initial temperature for annealing.
            cooling_rate: Cooling rate for temperature decay (0 < cooling_rate < 1).
            min_temp: Minimum temperature.

        """
        self.params = list(params)
        self.temperature = init_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.state = [p.clone() for p in self.params]

    def perturb(self):
        """Perturb parameters based on the current temperature."""
        with torch.no_grad():
            for param in self.params:
                noise = torch.randn_like(param) * self.temperature
                param.add_(noise)

    def step(self, eval_func: Callable[[], float]):
        """Perform one optimization step using Simulated Annealing.

        Args:
            eval_func: A function that evaluates the current loss.

        """
        # Save current state
        current_loss = eval_func()
        current_params = [p.clone() for p in self.params]

        # Apply perturbation
        self.perturb()
        new_loss = eval_func()

        if new_loss < current_loss:
            # Accept the new state
            self.state = [p.clone() for p in self.params]
        else:
            # Accept worse states with a certain probability
            acceptance_probability = torch.exp(
                torch.tensor((current_loss - new_loss) / self.temperature),
            ).item()
            if torch.rand(1).item() > acceptance_probability:
                # Revert to the previous state
                for param, saved in zip(self.params, current_params):
                    param.copy_(saved)

        # Cool down the temperature
        self.temperature = max(self.min_temp, self.temperature * self.cooling_rate)
