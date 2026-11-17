"""Generate deterministic synthetic users without inventing dataset metadata."""
import numpy as np
from simulation.users.profile import UserProfile


class UserGenerator:
    BEHAVIORS = ("active_viewer", "active_rater", "explorer", "preference_drifter")

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, count: int) -> list[UserProfile]:
        if count < 1:
            raise ValueError("count must be at least 1")
        users = []
        for index in range(count):
            behavior = self.BEHAVIORS[index % len(self.BEHAVIORS)]
            vector = self.rng.normal(0.0, 0.7, size=4)
            users.append(UserProfile(
                user_id=f"sim_user_{index + 1:03d}",
                behavior_type=behavior,
                activity_level=float(self.rng.uniform(0.35, 1.0)),
                exploration_rate=float(self.rng.uniform(0.15, 0.85)),
                rating_bias=float(self.rng.normal(0.0, 0.25)),
                preference_vector=tuple(float(x) for x in vector),
            ))
        return users
