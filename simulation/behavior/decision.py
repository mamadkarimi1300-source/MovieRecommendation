"""Create plausible synthetic decisions from seeded latent preferences."""
import hashlib
import numpy as np


class BehaviorEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.user_preferences = {}
        self.user_steps = {}

    @staticmethod
    def _item_vector(movie_title: str) -> np.ndarray:
        digest = hashlib.sha256(movie_title.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        return np.random.default_rng(seed).normal(0.0, 0.7, size=4)

    def _preference(self, user, allow_drift=False):
        vector = self.user_preferences.setdefault(
            user.user_id, np.asarray(user.preference_vector, dtype=float).copy()
        )
        if allow_drift and user.behavior_type == "preference_drifter":
            step = self.user_steps.get(user.user_id, 0) + 1
            self.user_steps[user.user_id] = step
            if step % 2 == 0:
                vector = 0.88 * vector + self.rng.normal(0.0, 0.16, size=4)
                self.user_preferences[user.user_id] = vector
        return vector

    def choose_movie(self, user, catalog: list[str], seen: set[str]) -> str:
        candidates = [title for title in catalog if title not in seen]
        if not candidates:
            candidates = catalog
        preference = self._preference(user, allow_drift=True)
        if self.rng.random() < user.exploration_rate:
            return str(self.rng.choice(candidates))
        scores = [float(np.dot(preference, self._item_vector(title))) for title in candidates]
        return candidates[int(np.argmax(scores))]

    def make_rating(self, user, movie_title: str) -> float:
        preference = self._preference(user)
        affinity = float(np.dot(preference, self._item_vector(movie_title)))
        raw = 3.25 + 0.48 * affinity + user.rating_bias + float(self.rng.normal(0, 0.65))
        return round(float(np.clip(raw, 1.0, 5.0)) * 2) / 2
