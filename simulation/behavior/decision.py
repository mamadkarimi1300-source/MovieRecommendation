"""Create plausible synthetic rating decisions from seeded latent preferences."""
import hashlib
import numpy as np


class BehaviorEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _item_vector(movie_title: str) -> np.ndarray:
        digest = hashlib.sha256(movie_title.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        return np.random.default_rng(seed).normal(0.0, 0.7, size=4)

    def choose_movie(self, user, catalog: list[str], seen: set[str]) -> str:
        candidates = [title for title in catalog if title not in seen]
        if not candidates:
            candidates = catalog
        if user.behavior_type == "explorer" and self.rng.random() < user.exploration_rate:
            return str(self.rng.choice(candidates))
        scores = [float(np.dot(user.preference_vector, self._item_vector(title))) for title in candidates]
        if self.rng.random() < user.exploration_rate:
            return str(self.rng.choice(candidates))
        return candidates[int(np.argmax(scores))]

    def make_rating(self, user, movie_title: str) -> float:
        affinity = float(np.dot(user.preference_vector, self._item_vector(movie_title)))
        raw = 3.25 + 0.48 * affinity + user.rating_bias + float(self.rng.normal(0, 0.65))
        return round(float(np.clip(raw, 1.0, 5.0)) * 2) / 2
