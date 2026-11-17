# Movie Recommendation System

A compact machine-learning movie recommender built with the MovieLens 100K ratings dataset and SVD.

## Features
- Loads the full MovieLens 100K ratings dataset through TensorFlow Datasets.
- Splits ratings into 50% training and 50% testing data (fixed random seed for reproducibility).
- Trains an SVD collaborative-filtering model using training data only.
- Evaluates predictions on held-out test ratings with RMSE and MAE.
- Provides a Persian RTL Streamlit dashboard with dataset analysis, step-by-step training status, detailed evaluation, personalized recommendations, and downloadable reports.
- Recommends unseen movies for a selected user and exports results to CSV.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the web UI (Windows)

```powershell
streamlit run app.py
```

Streamlit prints a local URL (usually http://localhost:8501) to open in your browser. Use the **Training & Evaluation** tab to download/load the dataset, train SVD, evaluate on the held-out 50% test split, and save the model. The **Movie Recommendations** tab uses the trained or saved model.

### Dashboard sections and training outputs

- **Overview:** dataset totals and the latest RMSE/MAE results.
- **Data analysis:** rating distribution, most active users, most-rated movies, and a data preview.
- **Training & evaluation:** select the SVD latent-component count, follow the five training stages, and inspect matrix dimensions, matrix density, explained variance ratio, elapsed time, and fallback prediction counts.
- **Personalized recommendations:** choose a user, inspect their training ratings, rank unseen movies, and export recommendations to CSV.
- **Reports & exports:** download the training report as JSON and per-test-rating predictions as CSV. The evaluation CSV includes the actual rating, predicted rating, absolute error, signed error, and prediction source.

After a successful run, these files are written to `artifacts/`:
- `movie_recommender.joblib` — trained model.
- `training_report.json` — dataset, split, model, evaluation, and runtime metadata.
- `evaluation_predictions.csv` — row-level test predictions and errors.

The dashboard restores the saved report and evaluation table when it is restarted, provided the artifact files remain in the project folder.

## Run from the command line

Train, evaluate on the 50% test split, and recommend for user 1:
```bash
python main.py --train --user 1
```

Train and evaluate only:
```bash
python main.py --train
```

Get recommendations using the previously saved model:
```bash
python main.py --user 1
```

The first run downloads the dataset. Model and CSV output are stored in `artifacts/`.

## Evaluation protocol

The data is randomly divided into two disjoint parts using `random_state=42`:
- 50% training: used to build the user-item matrix and fit SVD.
- 50% testing: held out from model fitting and used only for RMSE/MAE evaluation.

Unknown users or movies in the test set receive the training-set global mean as a fallback prediction. This is a simple baseline fallback, not a personalized estimate. The random rating-level split is suitable for a compact demonstration; a time-aware or user-wise split would be more appropriate for some research questions.

## Project structure

```text
MovieRecommendation/
├── data/
│   └── dataset.py
├── recommender/
│   ├── model.py
│   └── recommend.py
├── artifacts/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## Notes
- The model uses user ID, movie title, and rating from MovieLens 100K.
- Recommendations exclude movies already rated by the selected user in the training data.
