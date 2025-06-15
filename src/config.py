"""Experiment configuration."""

# General
BATCH_SIZE = 16
POLICY_UPDATE_FREQUENCY = 16  # optimize every x updates
TARGET_UPDATE_FREQUENCY = 128  # update target net every x updates
MAX_INTERACTIONS = 64000  # training budget

# Genetic Algorithm
POPULATION_SIZE = 50
NUM_GENERATIONS = 20
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
ELITE_SIZE = 4  # number of elite individuals to keep


# Replay Memory sampling (prioritized true or false)

# further optimizers

# further options:
# ENV_NAME, SEED, NUM_EPOCHS,
# GAMMA, MAX_INTERACTIONS, REPLAY_CAPACITY,
# OPTIMIZER,
# VQC: N_LAYERS,
# EVAL_EPISODES = 25
