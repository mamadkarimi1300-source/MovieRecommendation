"""Load and prepare the MovieLens 100K ratings dataset."""

import pandas as pd
import tensorflow_datasets as tfds


def load_ratings() -> pd.DataFrame:
    """Download (on first run) and load all MovieLens 100K ratings."""
    dataset = tfds.load("movielens/100k-ratings", split="train")
    rows = list(dataset.as_numpy_iterator())
    frame = pd.DataFrame(rows)

    # Convert TensorFlow byte-string fields into normal Python strings.
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )

    required = {"user_id", "movie_title", "user_rating"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")

    frame = frame[["user_id", "movie_title", "user_rating"]].copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["movie_title"] = frame["movie_title"].astype(str).str.strip()
    frame["user_rating"] = pd.to_numeric(frame["user_rating"], errors="coerce")
    frame = frame.dropna().drop_duplicates(
        subset=["user_id", "movie_title"], keep="last"
    )
    return frame.reset_index(drop=True)
