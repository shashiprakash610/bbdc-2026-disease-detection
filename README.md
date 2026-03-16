# BBDC 2026 Disease Detection

Starter repository for the Bremen Big Data Challenge 2026 Student Track.

## Goal
Build a machine learning pipeline to predict `impairment_type` from physiological signals and provided features.

## Project structure

```text
bbdc-2026-disease-detection/
├── .vscode/                 # VS Code workspace settings
├── data/                    # competition data (not committed)
├── models/                  # trained model artifacts (not committed)
├── notebooks/               # exploration notebooks
├── src/                     # training and inference code
├── submissions/             # generated submission CSV files
├── .gitignore
├── README.md
└── requirements.txt
```

## Recommended workflow

1. Download the challenge dataset.
2. Put raw files under `data/`.
3. Start with the provided `features.csv` files.
4. Train a baseline model with `src/train_baseline.py`.
5. Generate a submission with `src/predict.py`.
6. Commit code and documentation to GitHub, but never the challenge data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

## Run baseline

```bash
python -m src.train_baseline
```

## Generate submission

```bash
python -m src.predict
```

## Notes

- Keep `data/` out of Git.
- Keep `models/` out of Git.
- Add your score progression and approach to this README so the repo becomes portfolio-quality.
