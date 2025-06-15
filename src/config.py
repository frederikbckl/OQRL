"""Experiment configuration."""

# General
BATCH_SIZE = 64
POLICY_UPDATE_FREQUENCY = 64  # optimize every x updates
TARGET_UPDATE_FREQUENCY = 512  # update target net every x updates

# Genetic Algorithm
POPULATION_SIZE = 25
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5


# Replay Memory sampling (prioritized true or false)

# further optimizers

# further options:
# ENV_NAME, SEED, NUM_EPOCHS,
# GAMMA, MAX_INTERACTIONS, REPLAY_CAPACITY,
# OPTIMIZER,
# VQC: N_LAYERS,
# EVAL_EPISODES = 25
