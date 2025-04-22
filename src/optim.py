import random

import numpy as np
import torch

from utils import Experience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAOptimizer:
    def __init__(
        self,
        model,
        population_size=20,
        num_generations=10,
        mutation_rate=0.1,
        crossover_rate=0.5,
    ):
        self.model = model
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Initialize population
        self.population = [self._initialize_individual() for _ in range(population_size)]
        self.best_individual = None

    def _initialize_individual(self):
        """Initialize an individual with random weights."""
        return [
            param.data.clone() + 0.1 * torch.randn_like(param) for param in self.model.parameters()
        ]

    def _evaluate_fitness(self, individual, loss_fn, batch):
        """Evaluate the fitness of an individual."""
        # Load the individual's weights into the model
        for param, ind_param in zip(self.model.parameters(), individual):
            param.data.copy_(ind_param)

        # print(f"Batch type: {type(batch)}")
        # print(f"Batch[0] type: {type(batch[0]) if isinstance(batch, (list, tuple)) else 'N/A'}")
        # print(f"Batch length: {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")

        # Check if batch is a list or tuple
        # print(f"Batch content: {batch}")
        if isinstance(batch[0], Experience):
            # Unpack Experience objects if present
            states = np.array([exp.obs for exp in batch])
            actions = np.array([exp.action for exp in batch])
            rewards = np.array([exp.reward for exp in batch])
            next_states = np.array([exp.next_obs for exp in batch])
            terminals = np.array([exp.terminated for exp in batch])
        else:
            # states, actions, rewards, next_states, terminals = batch
            states, actions, rewards, next_states, terminals = zip(*batch)

        # Unpack batch assuming it's a tuple of NumPy arrays or PyTorch tensors
        # states, actions, rewards, next_states, terminals = batch

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

        # if isinstance(batch[0], np.ndarray):  # If batch contains NumPy arrays directly
        #     states, actions, rewards, next_states, terminals = batch
        # else:  # If batch contains Experience objects
        #     states = np.array([exp.obs for exp in batch])
        #     actions = np.array([exp.action for exp in batch])
        #     rewards = np.array([exp.reward for exp in batch])
        #     next_states = np.array([exp.next_obs for exp in batch])
        #     terminals = np.array([exp.done for exp in batch])

        # Convert to PyTorch tensors
        # states = torch.tensor(states, dtype=torch.float32).to(device)
        # actions = torch.tensor(actions, dtype=torch.int64).to(device)
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        # terminals = torch.tensor(terminals, dtype=torch.float32).to(device)

        # Convert to PyTorch tensors

        # NEW
        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.from_numpy(np.array(actions)).long().to(device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        terminals = torch.from_numpy(np.array(terminals)).float().to(device)

        # OLD
        # states = torch.tensor(states, dtype=torch.float32).to(device)
        # actions = torch.tensor(actions, dtype=torch.int64).to(device)
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        # terminals = torch.tensor(terminals, dtype=torch.float32).to(device)

        q_values = (
            self.model(states).gather(1, actions.view(-1, 1)).squeeze()
        )  # Q-values for selected actions
        next_q_values = self.model(next_states).max(1)[0].detach()  # Max Q-value for next state
        targets = rewards + (1 - terminals) * 0.99 * next_q_values  # Bellman equation

        loss = torch.nn.functional.mse_loss(q_values, targets)  # Compute MSE loss
        # print(f"Fitness computed: {-loss.item()}")  # Debugging
        return -loss.item()  # Negative loss as fitness (to maximize reward)

    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        child1, child2 = [], []
        for p1, p2 in zip(parent1, parent2):
            mask = torch.rand_like(p1) < self.crossover_rate
            child1.append(torch.where(mask, p1, p2))
            child2.append(torch.where(mask, p2, p1))
        return child1, child2

    def _mutate(self, individual):
        """Mutate an individual by adding random noise."""
        for param in individual:
            if random.random() < self.mutation_rate:
                param += 0.1 * torch.randn_like(param)

    def optimize(self, loss_fn, batch):
        """Run the genetic algorithm optimization."""
        for generation in range(self.num_generations):
            print(f"GAOptimizer: Generation {generation+1}/{self.num_generations}")
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
                # Select parents
                parent1, parent2 = random.sample(self.population[:5], 2)  # random selection (old)
                # parent1, parent2 = [
                #     self.tournament_selection(loss_fn, batch, k=5) for _ in range(2)
                # ]  # Tournament selection

                # Crossover and mutation
                child1, child2 = self._crossover(parent1, parent2)
                self._mutate(child1)
                self._mutate(child2)

                # Add children to the next generation
                next_population.extend([child1, child2])

            self.population = next_population[: self.population_size]

        # delete this log later
        print("Finished GA optimization for batch")

        # Load the best individual's weights into the model
        if self.best_individual is not None:
            for param, best_param in zip(self.model.parameters(), self.best_individual):
                param.data.copy_(best_param)
        else:
            print("Warning: best_individual is None. Skipping weight update.")

    def tournament_selection(self, loss_fn, batch, k=7):
        """Select the best individual out of k randomly chosen ones."""
        selected = random.sample(self.population, k)
        return min(selected, key=lambda ind: self._evaluate_fitness(ind, loss_fn, batch))

    # def elitism(self, elite_size=2):
    #     """Preserve the best individuals."""
    #     sorted_population = sorted(
    #         self.population,
    #         key=lambda ind: self._evaluate_fitness(ind, loss_fn, batch),
    #     )
    #     return sorted_population[:elite_size]  # Keep the best ones


# class SimulatedAnnealing:
#     """Simulated Annealing Optimizer."""

#     def __init__(
#         self,
#         params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
#         init_temp: float,
#         cooling_rate: float,
#         min_temp: float,
#     ):
#         """Initialize the SimulatedAnnealing optimizer.

#         Args:
#             params: Parameters to optimize.
#             init_temp: Initial temperature for annealing.
#             cooling_rate: Cooling rate for temperature decay (0 < cooling_rate < 1).
#             min_temp: Minimum temperature.

#         """
#         self.params = list(params)
#         self.temperature = init_temp
#         self.cooling_rate = cooling_rate
#         self.min_temp = min_temp
#         self.state = [p.clone() for p in self.params]

#     def perturb(self):
#         """Perturb parameters based on the current temperature."""
#         with torch.no_grad():
#             for param in self.params:
#                 noise = torch.randn_like(param) * self.temperature
#                 param.add_(noise)

#     def step(self, eval_func: Callable[[], float]):
#         """Perform one optimization step using Simulated Annealing.

#         Args:
#             eval_func: A function that evaluates the current loss.

#         """
#         # Save current state
#         current_loss = eval_func()
#         current_params = [p.clone() for p in self.params]

#         # Apply perturbation
#         self.perturb()
#         new_loss = eval_func()

#         if new_loss < current_loss:
#             # Accept the new state
#             self.state = [p.clone() for p in self.params]
#         else:
#             # Accept worse states with a certain probability
#             acceptance_probability = torch.exp(
#                 torch.tensor((current_loss - new_loss) / self.temperature),
#             ).item()
#             if torch.rand(1).item() > acceptance_probability:
#                 # Revert to the previous state
#                 for param, saved in zip(self.params, current_params):
#                     param.copy_(saved)

#         # Cool down the temperature
#         self.temperature = max(self.min_temp, self.temperature * self.cooling_rate)
