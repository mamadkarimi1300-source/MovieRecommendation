# Movie Recommendation System

A MovieLens 100K recommendation project with a Persian RTL dashboard, a synthetic-user simulation environment, and an incremental-learning laboratory.

## Features

- Loads MovieLens 100K ratings and trains the existing TruncatedSVD recommender.
- Reproducible 50% training / 50% test split with RMSE and MAE evaluation.
- Recommender workspace: train SVD, inspect user history, evaluate held-out ratings, and generate top-N recommendations.
- Simulation workspace: generate 50 synthetic users with four behavior patterns and emit timestamped movie-view, movie-rating, recommendation-view, and recommendation-click events.
- Laboratory workspace: consume simulation events, deduplicate event IDs, run prequential evaluation, and incrementally update a separate matrix-factorization model.
- Persist experiment summaries as JSON under `artifacts/experiments/history.json`.

## Setup (Windows)

```powershell
py -m venv .venv
.venv\\Scripts\\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Run the dashboard

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

## Workspaces

### Recommender

The original SVD workflow remains separate and unchanged: load MovieLens 100K, split ratings, train TruncatedSVD, evaluate held-out ratings, save the model, and generate recommendations for known dataset users.

### Simulation

The initial simulation uses 50 generated users and a configurable number of interactions per user. Profiles rotate through active viewer, active rater, explorer, and preference-drifter behavior labels. A seeded latent-preference policy selects movie titles from the real MovieLens catalog and generates synthetic ratings in the 1–5 range.

Synthetic ratings are simulated data, not MovieLens ground truth. The dataset does not provide genre metadata to this simulator, so no genre preferences are fabricated.

When an SVD model has been trained, the simulator can also expose synthetic users to a candidate list derived from the model's aggregate movie scores. Synthetic user IDs are not present in the trained SVD user matrix, so this first version does not claim personalized SVD predictions for those cold-start users.

### Laboratory

Simulation events are passed directly to the laboratory engine in-process. The laboratory validates event IDs and processes each event once. Rating events are evaluated before the same rating is used to update the incremental matrix-factorization model (prequential evaluation).

The incremental model is independent of the existing SVD model. It uses user/movie latent factors and bias terms and updates online after each rating. RMSE and MAE shown in the laboratory are cumulative prequential metrics over the processed synthetic ratings.

Experiment summaries can be saved to `artifacts/experiments/history.json`. The current in-memory model and event de-duplication set live for the Streamlit session; they are not yet checkpointed for recovery across app restarts.

## Tests

Run the focused simulation and laboratory tests:

```powershell
py -m unittest discover -s tests -v
```

## Existing command-line workflow

```powershell
py main.py --train --user 1
```

The original CLI and SVD model remain available.
