"""Data model for a synthetic user."""
from dataclasses import dataclass


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    behavior_type: str
    activity_level: float
    exploration_rate: float
    rating_bias: float
    preference_vector: tuple[float, ...]
