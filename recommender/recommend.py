"""Generate personalized top-N movie recommendations."""

import pandas as pd

from recommender.model import MovieRecommender


def recommend_movies(
    model: MovieRecommender, user_id: str | int, top_n: int = 10
) -> pd.DataFrame:
    user_id = str(user_id)

    if model.user_item is None or model.predicted_ratings is None:
        raise RuntimeError("The model has not been trained.")

    if user_id not in model.user_item.index:
        raise ValueError(f"User {user_id!r} was not found in the training data.")

    rated_titles = set(
        model.user_item.loc[user_id].dropna().index.astype(str)
    )
    candidates = model.predicted_ratings.loc[user_id].drop(
        labels=list(rated_titles), errors="ignore"
    )

    result = candidates.nlargest(top_n).rename("predicted_rating").reset_index()
    result.columns = ["movie_title", "predicted_rating"]
    result.insert(0, "user_id", user_id)
    return result
