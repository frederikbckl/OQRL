# import random

from abc import ABC, abstractmethod

import numpy as np
import torch

from utils import device


class BaseOptimizer(ABC):
    def __init__(self, model, rng=None):
        self.model = model
        self.rng = rng or np.random.default_rng()  # Use seeded RNG or fallback if not provided

    @abstractmethod
    def optimize(self, loss_fn, batch):
        """Run full optimization loop."""


class GAOptimizer(BaseOptimizer):
    def __init__(
        self,
        model,
        population_size=25,
        num_generations=10,
        mutation_rate=0.1,
        crossover_rate=0.5,
        rng=None,
    ):
        super().__init__(model, rng)
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = rng or np.random.default_rng()  # Use seeded RNG or fallback if not provided

        # self.total_interactions = 0  # calculated interactions

        # Initialize population
        self.population = [self._initialize_individual() for _ in range(population_size)]
        self.best_individual = None

        self.interaction_count = 0  # actually counted interactions

    def _initialize_individual(self):
        """Initialize an individual with random weights."""
        # new seeded initialization
        individual = []
        for param in self.model.parameters():
            shape = param.shape
            # Use numpy RNG to sample normal distribution and convert to tensor
            noise = self.rng.normal(loc=0.0, scale=1.0, size=shape)
            noise_tensor = torch.tensor(noise, dtype=torch.float32, device=param.device)
            individual.append(param.data.clone() + 0.1 * noise_tensor)
        return individual

    # NEW _evaluate_fitness
    def _evaluate_fitness(self, individual, loss_fn, batch):
        """Evaluate the fitness of an individual."""
        # Load the individual's weights into the model (without these 2 lines, the agent is not learning)
        # without it, the model would retain the weights of the previous generation (must be updated before each fitness evaluation)
        # ensures that each fitness evaluation uses the correct weights for the model
        for param, ind_param in zip(self.model.parameters(), individual):
            param.data.copy_(ind_param)

        # Unpack Experience objects directly
        states = torch.stack(
            [
                exp.obs
                if isinstance(exp.obs, torch.Tensor)
                else torch.tensor(exp.obs, dtype=torch.float32)
                for exp in batch
            ],
        ).to(device)
        actions = torch.stack(
            [
                exp.action
                if isinstance(exp.action, torch.Tensor)
                else torch.tensor(exp.action, dtype=torch.int64)
                for exp in batch
            ],
        ).to(device)
        rewards = torch.stack(
            [
                exp.reward
                if isinstance(exp.reward, torch.Tensor)
                else torch.tensor(exp.reward, dtype=torch.float32)
                for exp in batch
            ],
        ).to(device)
        next_states = torch.stack(
            [
                exp.next_obs
                if isinstance(exp.next_obs, torch.Tensor)
                else torch.tensor(exp.next_obs, dtype=torch.float32)
                for exp in batch
            ],
        ).to(device)
        terminals = torch.stack(
            [
                exp.terminated
                if isinstance(exp.terminated, torch.Tensor)
                else torch.tensor(exp.terminated, dtype=torch.float32)
                for exp in batch
            ],
        ).to(device)

        # Convert to numpy after moving to CPU
        states_cpu = states.cpu().numpy()
        actions_cpu = actions.cpu().numpy()
        rewards_cpu = rewards.cpu().numpy()
        next_states_cpu = next_states.cpu().numpy()
        terminals_cpu = terminals.cpu().numpy()

        # Check if data is already a tensor, if so, use .clone().detach(), otherwise convert
        states = (
            states.clone().detach().to(device)
            if isinstance(states, torch.Tensor)
            else torch.tensor(states, dtype=torch.float32).to(device)
        )
        actions = (
            actions.clone().detach().to(device)
            if isinstance(actions, torch.Tensor)
            else torch.tensor(actions, dtype=torch.int64).to(device)
        )
        rewards = (
            rewards.clone().detach().to(device)
            if isinstance(rewards, torch.Tensor)
            else torch.tensor(rewards, dtype=torch.float32).to(device)
        )
        next_states = (
            next_states.clone().detach().to(device)
            if isinstance(next_states, torch.Tensor)
            else torch.tensor(next_states, dtype=torch.float32).to(device)
        )
        terminals = (
            terminals.clone().detach().to(device)
            if isinstance(terminals, torch.Tensor)
            else torch.tensor(terminals, dtype=torch.float32).to(device)
        )

        # ensure actions and states are both on the same device before computing q_values
        states = states.to(device)
        actions = actions.to(device)

        model_output = self.model(states.to(device))
        model_output = model_output.to(device)  # Ensure model output is on the same device

        if model_output.device != actions.device:
            model_output = model_output.to(actions.device)

        q_values = model_output.gather(
            1,
            actions.view(-1, 1),
        ).squeeze()  # Q-values for selected actions
        next_q_values = self.model(next_states).max(1)[0].detach()  # Max Q-value for next state
        targets = rewards.to(device) + (1 - terminals.to(device)) * 0.99 * next_q_values.to(
            device,
        )  # Bellman equation

        loss = torch.nn.functional.mse_loss(q_values, targets)  # Compute MSE loss
        # print(f"Fitness computed: {-loss.item()}")  # Debugging

        # count interactions with the offline dataset
        self.interaction_count += len(batch)  # or should it be incremented by len(batch)?

        return -loss.item()  # Negative loss as fitness (to maximize reward)

    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        child1, child2 = [], []
        for p1, p2 in zip(parent1, parent2):
            # new mask: seeded rng
            shape = p1.shape
            # Generate deterministic mask using seeded RNG
            mask_np = self.rng.random(shape) < self.crossover_rate
            mask = torch.tensor(mask_np, dtype=torch.bool, device=p1.device)

            child1.append(torch.where(mask, p1, p2))
            child2.append(torch.where(mask, p2, p1))
        return child1, child2

    def _mutate(self, individual):
        """Mutate an individual by adding noise."""
        for param in individual:
            with torch.no_grad():
                noise = torch.tensor(
                    self.rng.normal(0, 0.1, size=param.shape),
                    device=param.device,
                )
                param.add_(noise)

    def optimize(self, loss_fn, batch):
        """Run the genetic algorithm optimization."""
        for generation in range(self.num_generations):
            # print(f"GAOptimizer: Generation {generation+1}/{self.num_generations}")
            # Evaluate fitness for each individual
            fitness = [self._evaluate_fitness(ind, loss_fn, batch) for ind in self.population]

            # Select individuals based on fitness (elitism)
            sorted_indices = np.argsort(fitness)[::-1]
            self.population = [self.population[i] for i in sorted_indices]
            self.best_individual = self.population[0]

            # Generate next generation
            next_population = self.population[:2]
            # next_population = sorted(
            #     self.population,
            #     key=lambda ind: self._evaluate_fitness(ind, loss_fn, batch),
            # )[:2]  # Elitism: retain the top 2 individuals

            while len(next_population) < self.population_size:
                # New seeded parent selection
                parent_indices = self.rng.choice(5, size=2, replace=False)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]

                # Crossover and mutation
                child1, child2 = self._crossover(parent1, parent2)
                self._mutate(child1)
                self._mutate(child2)

                # Add children to the next generation
                next_population.extend([child1, child2])

            self.population = next_population[: self.population_size]

        # Load the best individual's weights into the model
        if self.best_individual is not None:
            for param, best_param in zip(self.model.parameters(), self.best_individual):
                param.data.copy_(best_param)
        else:
            print("Warning: best_individual is None. Skipping weight update.")

    # def elitism(self, elite_size=2):
    #     """Preserve the best individuals."""
    #     sorted_population = sorted(
    #         self.population,
    #         key=lambda ind: self._evaluate_fitness(ind, loss_fn, batch),
    #     )
    #     return sorted_population[:elite_size]  # Keep the best ones
