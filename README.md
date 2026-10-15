# Predicting Bridge Performance under Earthquakes with Supervised Learning

UE24CS352A – Machine Learning · Mini-Project

Binary classification: given a bridge's **age**, the earthquake **magnitude** and the **distance from the epicenter**, predict whether the bridge was **damaged (1)** or **undamaged (0)**.
Three classifiers are compared: **Logistic Regression**, **Quadratic Discriminant Analysis (QDA)** and **K-Nearest Neighbours (KNN)**. Because damaged bridges are rare (~1 in 11 records), the positive class is up-sampled with **bootstrap resampling**.

## Source and attribution
The dataset, code and reported results come from the public Stanford CS229 (Fall 2018) project by Ziyang Jiang and Xiao Zhang:
https://github.com/jzy95310/CS229-Fall-2018-Final-Report
This repository is a cleaned, documented copy prepared for the mini-project (removed `__pycache__` and the duplicate zip, added this README, `requirements.txt` and `run_all.sh`). The algorithms and data are unchanged.

## Repository layout
```
bridge-earthquake-ml/
├── README.md
├── requirements.txt
├── bridge_earthquake_analysis.ipynb   # Jupyter notebook: data, models, plots, K selection (run from this folder)
├── run_all.sh              # runs the whole pipeline
├── Code/
│   ├── util.py             # CSV loader (columns starting with x = features, y = label)
│   ├── linear_model.py     # base class used by util
│   ├── plot_data.py        # scatter plot of magnitude vs distance -> Output/data.png
│   ├── KNN_optimal.py      # picks K (1..10) on a validation set -> Output/KNN_optimal.png
│   ├── predict_x2_x4_x5.py # MAIN: 3 features (age, magnitude, distance), LR + QDA + KNN
│   ├── predict_x4_x5.py    # 2 features (magnitude, distance), LR + QDA + KNN
│   └── prelim_code/        # early LR / GDA visualisation experiments
├── Data/
│   ├── ds2.csv                              # master table: 249 bridge-earthquake records (21 damaged)
│   ├── ds2_x2_x4_x5_train.csv / _test.csv   # 199 train (18 pos) / 50 test (3 pos)
│   ├── ds2_x4_x5_train.csv / _test.csv      # 2-feature version
│   ├── KNN_optimal/ds2_knn_train.csv, ds2_knn_cv.csv   # 149 train / 50 validation for choosing K
│   └── ds1*.csv, ds1_original.xlsx          # earlier smaller dataset (preliminary)
└── Output/                 # plots + accuracy tables (committed results)
```
Feature columns: `x2` = bridge age (years), `x4` = earthquake magnitude, `x5` = distance to epicenter (miles), label `y1` = damaged (1) / not (0).
(`x1` material type and `x3` bridge type exist in `ds2.csv` but are not used by the models.)

## Setup
Python 3.8+ is required.
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
Scripts use relative paths (`../Data`, `../Output`), so run them **from inside `Code/`**:
```bash
cd Code
python plot_data.py               # data scatter plot
python KNN_optimal.py             # prints "The optimal K value is 2."  (~1 min)
python predict_x2_x4_x5.py        # MAIN experiment: prints train/test accuracy for LR, QDA, KNN (~10 s)
python predict_x4_x5.py           # optional 2-feature experiment
```
or simply `bash run_all.sh` from the repository root.

**Notebook:** `pip install notebook pandas`, then `jupyter notebook bridge_earthquake_analysis.ipynb` (launch from the repository root so `Data/` is found) and choose *Run All*.
No display is needed: set `MPLBACKEND=Agg` if running on a headless machine (`run_all.sh` does this).

## Expected results (mean of 100 bootstrap iterations x 5 positive:negative ratios 0.2-1.0)
| Model | Train acc. | Test acc. | Test acc. (damaged) | Test acc. (undamaged) |
|---|---|---|---|---|
| Logistic Regression | ~95.4% | ~74% | ~83% | ~73% |
| **QDA** | ~97.8% | **~94.8%** | ~99.5% | ~94.5% |
| KNN (K=2) | ~99.1% | ~80.6% | ~30% | ~84% |

Bootstrap resampling is random, so re-running changes the numbers by a few tenths of a percent.
The test set contains only 3 damaged bridges, so the per-class "damaged" accuracy is very coarse; see the write-up for discussion.
