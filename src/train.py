"""Simplified training module."""

import gymnasium as gym


def run_train(agent_fac, env_name, num_epochs, seed):
    """Run a very basic training loop."""
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Failed to create environment {env_name}: {e}")
        raise

    act_space = env.action_space  # Get the action space

    agent = agent_fac.create(obs_space, act_space, seed)  # Create the agent instance

    obs = env.reset()
    action = agent.policy(obs)  # Assuming your agent's policy method is correctly implemented
    print(env.step(action))  # This print statement is to debug what env.step returns


# This updated version explicitly creates an agent from the factory before the training loop.
