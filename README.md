# Movie Recommendation System

A compact machine-learning movie recommender built with the MovieLens 100K ratings dataset and SVD.

## Features
- Loads the full MovieLens 100K ratings dataset through TensorFlow Datasets.
- Trains an SVD collaborative-filtering model.
- Evaluates predictions with RMSE and MAE.
- Recommends the top 10 unseen movies for a selected user.
- Exports recommendations to CSV.

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

## Run

Train and evaluate the model:
```bash
python main.py --train
```

Get recommendations for user 1 (uses the saved model; train first):
```bash
python main.py --user 1
```

Train and recommend in one command:
```bash
python main.py --train --user 1
```

The first run downloads the dataset. Model and CSV output are stored in `artifacts/`.

## Project structure

```text
MovieRecommendation/
├── data/
│   └── dataset.py
├── recommender/
│   ├── model.py
│   └── recommend.py
├── artifacts/
├── main.py
├── requirements.txt
└── README.md
```

## Notes
- The model uses user ID, movie title, and rating from MovieLens 100K.
- A random train/test split is used for a simple demonstration. For rigorous research, use a time-aware split.
- Recommendations exclude movies already rated by the selected user.
