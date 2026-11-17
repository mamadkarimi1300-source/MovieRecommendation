"""A lightweight SVD-based collaborative filtering model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


class MovieRecommender:
    def __init__(self, n_components: int = 20, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = TruncatedSVD(
            n_components=n_components, random_state=random_state
        )
        self.user_item = None
        self.predicted_ratings = None
        self.global_mean = 0.0

    def fit(self, ratings: pd.DataFrame):
        """Fit latent factors using a user-item rating matrix."""
        self.global_mean = float(ratings["user_rating"].mean())
        self.user_item = ratings.pivot_table(
            index="user_id",
            columns="movie_title",
            values="user_rating",
            aggfunc="mean",
        )

        # Fill missing entries with each user's mean rating.
        user_means = self.user_item.mean(axis=1).fillna(self.global_mean)
        filled = self.user_item.T.fillna(user_means).T

        max_components = min(filled.shape) - 1
        if max_components < 1:
            raise ValueError("Not enough users or movies to train the model.")
        self.model = TruncatedSVD(
            n_components=min(self.n_components, max_components),
            random_state=self.random_state,
        )
        latent = self.model.fit_transform(filled)
        reconstructed = self.model.inverse_transform(latent)

        self.predicted_ratings = pd.DataFrame(
            reconstructed,
            index=self.user_item.index,
            columns=self.user_item.columns,
        ).clip(1.0, 5.0)
        return self

    def evaluate(self, test_ratings: pd.DataFrame) -> dict:
        """Calculate RMSE and MAE for held-out ratings."""
        actual, predicted = [], []
        for row in test_ratings.itertuples(index=False):
            user_id = str(row.user_id)
            title = row.movie_title
            if (
                self.predicted_ratings is not None
                and user_id in self.predicted_ratings.index
                and title in self.predicted_ratings.columns
            ):
                estimate = self.predicted_ratings.loc[user_id, title]
            else:
                estimate = self.global_mean
            actual.append(float(row.user_rating))
            predicted.append(float(estimate))

        errors = np.asarray(actual) - np.asarray(predicted)
        return {
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(np.abs(errors))),
            "test_count": len(actual),
        }

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path):
        return joblib.load(path)


def split_ratings(ratings: pd.DataFrame, test_size: float = 0.5):
    """Create reproducible 50/50 training and test sets by default."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    return train_test_split(
        ratings, test_size=test_size, random_state=42
    )
