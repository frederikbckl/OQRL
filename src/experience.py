from typing import Any, Dict, NamedTuple


class Experience(NamedTuple):
    """Represents an Experience."""

    obs: Any
    action: Any
    reward: Any
    next_obs: Any
    terminated: Any  # previously bool
    # truncated: bool
    info: Dict[str, Any]
