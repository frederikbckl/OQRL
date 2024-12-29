"""Simplified DQN module."""


class DQN:
    """Basic implementation of the Deep Q-Network algorithm."""

    def __init__(self, obs_space, act_space, rng, device="cpu"):
        """Initialize basic settings for DQN."""
        self.obs_space = obs_space
        self.act_space = act_space
        self.rng = rng
        self.device = device

    def policy(self, obs):
        """Simple policy to select an action."""
        return self.act_space.sample()  # Simplified to just sample an action

    def update(self, exp):
        """Placeholder for the update method."""
        pass  # No operation for now

    def on_step_end(self):
        """Placeholder method for actions at the end of each step."""
        pass

    def on_episode_end(self):
        """Placeholder method for actions at the end of each episode."""
        pass
