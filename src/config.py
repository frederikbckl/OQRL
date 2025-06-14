"""Experiment configuration."""

# General
BATCH_SIZE = 4
POLICY_UPDATE_FREQUENCY = 4  # optimize every x updates
TARGET_UPDATE_FREQUENCY = 16  # update target net every x updates

# Genetic Algorithm
POPULATION_SIZE = 25
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5

# further optimizers

# further options:
# ENV_NAME, SEED, NUM_EPOCHS,
# GAMMA, MAX_INTERACTIONS, REPLAY_CAPACITY,
# OPTIMIZER,
# VQC: N_LAYERS,
# EVAL_EPISODES = 25
