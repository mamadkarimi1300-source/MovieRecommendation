# Movie Recommendation System

A compact movie recommendation project using MovieLens 100K, collaborative filtering, and TruncatedSVD.

## Features

- Loads MovieLens 100K ratings.
- Reproducible 50% training / 50% test split (random state 42).
- Trains an SVD model and evaluates held-out ratings with RMSE and MAE.
- Persian right-to-left Streamlit interface with two focused sections:
  1. Train the model.
  2. Demonstrate personalization for one selected user.
- For the selected user, shows known training preferences, predictions for held-out ratings, per-user RMSE/MAE, and top-10 unseen movie recommendations.

## Setup (Windows)

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Run the dashboard

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

## How to demonstrate the project

1. Open **۱. آموزش مدل** and click **شروع آموزش مدل**.
2. Wait for the four clearly labeled stages to finish: load data, split data, train SVD, evaluate and save.
3. Review the overall RMSE and MAE.
4. Open **۲. آزمایش شخصی‌سازی برای یک کاربر** and select a user.
5. Show the user's high/low training ratings, then show predictions against that user's held-out actual ratings.
6. Click **تولید پیشنهادهای شخصی این کاربر** to display the top 10 movies the model has not seen that user rate in the training split.

## Generated files

After successful training, the `artifacts/` directory contains:

- `movie_recommender.joblib`: trained model.
- `test_ratings.csv`: held-out test ratings used for the single-user demonstration.
- `training_summary.json`: overall training and evaluation summary.

The model and test split are saved so the dashboard can be closed and reopened without retraining. Run training again if you want to regenerate these files.

## Evaluation notes

The split is performed at the individual-rating level, not by user or time. The selected user's held-out ratings are excluded from fitting and are used only after training to compare predicted and actual ratings. If a test movie was not present in the training matrix, the model uses the training global mean as a fallback; the UI labels that prediction accordingly.

The recommendation score is an estimated rating from the SVD reconstruction, not a probability that the user will like the movie.

## Command-line usage

```powershell
py main.py --train --user 1
```
