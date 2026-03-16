# BBDC 2026 – Disease Detection from Physiological Signals

Machine learning project for the Bremen Big Data Challenge 2026.

This project focuses on detecting age-related diseases using physiological signals such as EEG (brain activity), ECG (heart activity), and EDA (skin conductance).

The goal is to build machine learning models that can classify physiological states into healthy or diseased categories.

---

# Problem Statement

The dataset contains physiological signals collected while participants performed tasks in a virtual reality environment simulating different diseases.

Signals included:

* EEG – brain activity
* ECG – heart electrical activity
* EDA – electrodermal activity (skin conductance)

Each sample represents a 10-second signal segment.

The objective is to predict the disease class:

| Class | Description             |
| ----- | ----------------------- |
| 0     | Healthy                 |
| 1–6   | Different disease types |

Evaluation metric: **Macro F1 Score**

---

# Project Structure

```
bbdc-2026-disease-detection
│
├── data/                # Dataset (not tracked by Git)
├── notebooks/           # Exploratory analysis
├── src/
│   ├── data_loader.py
│   ├── train_baseline.py
│   ├── predict.py
│   └── config.py
│
├── models/              # Saved models
├── submissions/         # Competition submission files
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost / LightGBM
* Matplotlib

Development environment:

* VS Code
* Git
* GitHub

---

# Workflow

1. Download challenge dataset
2. Perform exploratory data analysis
3. Train baseline model using provided features
4. Improve performance through feature engineering
5. Train advanced models
6. Generate predictions for test data
7. Submit predictions to competition leaderboard

---

# Baseline Model

Initial experiments will use:

* Random Forest
* Gradient Boosting
* XGBoost

Baseline pipeline:

```
Feature Data → Train Model → Evaluate (Macro F1) → Generate Predictions
```

---

# Reproducing the Project

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/bbdc-2026-disease-detection.git
cd bbdc-2026-disease-detection
```

Create environment:

```
python -m venv .venv
```

Activate environment:

Mac/Linux

```
source .venv/bin/activate
```

Windows

```
.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Results

Results and leaderboard performance will be updated during the competition.

---

# Author

Shashi Prakash
Machine Learning / AI Engineer

---

# License

This repository is for educational and research purposes related to the Bremen Big Data Challenge.
