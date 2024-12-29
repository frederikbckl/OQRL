"""Simplified agent module."""


class Agent:
    """Basic agent class."""

    def __init__(self, obs_space, act_space, rng, device="cpu"):
        """Initialize the agent with minimal settings."""
        self.obs_space = obs_space
        self.act_space = act_space
        self.rng = rng
        self.device = device

    def policy(self, obs):
        """Placeholder for the policy to return an action."""
        return self.act_space.sample()  # Simplified to just sample an action

    def update(self, exp):
        """Placeholder for update method."""
        pass  # No operation for now


class AgentFactory:
    """Factory class to create agent instances."""

    def __init__(self, agent_class, **kwargs):
        """Initialize the factory with an agent class."""
        self.agent_class = agent_class
        self.kwargs = kwargs

    def create(self, obs_space, act_space, rng):
        """Create an agent instance."""
        return self.agent_class(obs_space, act_space, rng, **self.kwargs)
