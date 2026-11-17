"""Online matrix-factorization model with prequential predictions."""
import hashlib
import numpy as np


class IncrementalMatrixFactorization:
    def __init__(self, factors=12, learning_rate=0.015, regularization=0.02):
        self.factors = int(factors)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.user_factors = {}
        self.item_factors = {}
        self.user_bias = {}
        self.item_bias = {}
        self.global_mean = 3.5
        self.count = 0
        self.rating_sum = 0.0

    def _vector(self, key):
        digest = hashlib.sha256(str(key).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        return np.random.default_rng(seed).normal(0, 0.08, self.factors)

    def predict(self, user_id, movie_title):
        user_id, movie_title = str(user_id), str(movie_title)
        p, q = self.user_factors.get(user_id), self.item_factors.get(movie_title)
        if p is None or q is None:
            return float(np.clip(self.global_mean, 1, 5))
        value = self.global_mean + self.user_bias.get(user_id, 0.0) + self.item_bias.get(movie_title, 0.0) + float(np.dot(p, q))
        return float(np.clip(value, 1.0, 5.0))

    def update(self, user_id, movie_title, rating):
        user_id, movie_title, rating = str(user_id), str(movie_title), float(rating)
        if not 1 <= rating <= 5:
            raise ValueError("rating must be between 1 and 5")
        p = self.user_factors.setdefault(user_id, self._vector("user:" + user_id))
        q = self.item_factors.setdefault(movie_title, self._vector("item:" + movie_title))
        prediction = self.predict(user_id, movie_title)
        error = rating - prediction
        old_p = p.copy()
        lr, reg = self.learning_rate, self.regularization
        ub, ib = self.user_bias.get(user_id, 0.0), self.item_bias.get(movie_title, 0.0)
        self.user_bias[user_id] = ub + lr * (error - reg * ub)
        self.item_bias[movie_title] = ib + lr * (error - reg * ib)
        p += lr * (error * q - reg * p)
        q += lr * (error * old_p - reg * q)
        self.count += 1
        self.rating_sum += rating
        self.global_mean = self.rating_sum / self.count
        return prediction, error
