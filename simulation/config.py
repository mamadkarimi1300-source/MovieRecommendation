"""Configuration for reproducible synthetic-user simulations."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    user_count: int = 50
    interactions_per_user: int = 5
    random_seed: int = 42
    include_recommendation_events: bool = True

    def __post_init__(self):
        if self.user_count < 1:
            raise ValueError("user_count must be at least 1")
        if self.interactions_per_user < 1:
            raise ValueError("interactions_per_user must be at least 1")
