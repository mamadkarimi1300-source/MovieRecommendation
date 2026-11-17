"""Train, evaluate, and run the MovieLens recommendation demo."""

import argparse
from pathlib import Path

from data.dataset import load_ratings
from recommender.model import MovieRecommender, split_ratings
from recommender.recommend import recommend_movies

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "movie_recommender.joblib"


def train_model():
    print("Loading MovieLens 100K ratings...")
    ratings = load_ratings()
    print(f"Loaded {len(ratings):,} ratings from "
          f"{ratings['user_id'].nunique():,} users and "
          f"{ratings['movie_title'].nunique():,} movies.")

    train_data, test_data = split_ratings(ratings, test_size=0.5)
    print("\nData split")
    print(f"Training ratings: {len(train_data):,} (50%)")
    print(f"Testing ratings:  {len(test_data):,} (50%)")

    print("\nTraining SVD recommendation model...")
    model = MovieRecommender(n_components=20).fit(train_data)

    metrics = model.evaluate(test_data)
    print("\nEvaluation on unseen test ratings")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"Test ratings evaluated: {metrics['test_count']:,}")

    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Movie recommendations using MovieLens 100K and SVD."
    )
    parser.add_argument("--train", action="store_true",
                        help="Train and evaluate the model.")
    parser.add_argument("--user", type=str,
                        help="User ID for personalized recommendations.")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of recommendations (default: 10).")
    args = parser.parse_args()

    model = None
    if args.train:
        model = train_model()

    if args.user is not None:
        if model is None:
            if not MODEL_PATH.exists():
                parser.error("No saved model found. Run with --train first.")
            model = MovieRecommender.load(MODEL_PATH)

        recommendations = recommend_movies(model, args.user, args.top)
        print(f"\nTop {args.top} recommendations for user {args.user}:")
        if recommendations.empty:
            print("No unseen movies available for this user.")
        else:
            print(recommendations.to_string(index=False, formatters={
                "predicted_rating": "{:.2f}".format
            }))
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            output_path = ARTIFACTS / f"recommendations_user_{args.user}.csv"
            recommendations.to_csv(output_path, index=False)
            print(f"\nRecommendations saved to: {output_path}")

    if not args.train and args.user is None:
        parser.print_help()


if __name__ == "__main__":
    main()
