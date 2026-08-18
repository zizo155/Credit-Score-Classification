# Credit Score Classification

Multi-class classification of customer credit scores (Good / Standard /
Poor) from financial and behavioural features, with an interactive
Streamlit dashboard for exploration and model comparison.

The bulk of the work is data cleaning: the raw dataset is deliberately
messy, with corrupted numeric fields, placeholder strings, and
out-of-range values throughout.

## Dataset

[Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/)
from Kaggle — 100,000 training rows across 28 columns, plus 50,000
unlabelled test rows. After cleaning, 82,623 training rows remain.

Class distribution is imbalanced: Standard (44,042), Poor (23,966),
Good (14,615).

The CSVs are not included in this repository. Download them from the
link above and place `train.csv` and `test.csv` in the project root.

## Cleaning

The raw data required substantial repair before modelling:

- **Corrupted numerics** — `Annual_Income`, `Outstanding_Debt` and
  others carry stray underscores and symbols; stripped and coerced.
- **Placeholder strings** — `_______` in `Occupation`, `_` in
  `Credit_Mix`, and `!@9#%8` in `Payment_Behaviour` all mapped to
  "Not Specified".
- **Impossible values** — negative ages, ages above 75, and negative
  bank-account counts clipped or dropped.
- **Composite fields** — `Credit_History_Age` ("15 Years and 9 Months")
  parsed into a single month count.
- **Identifiers dropped** — `Name`, `SSN`, `ID`, `Customer_ID`.

Categorical columns are one-hot encoded, numeric columns standardised
with `StandardScaler`, and the target label-encoded.

## Results

Validation accuracy on a held-out 20% split:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 0.778 | 0.76 |
| Gradient Boosting | 0.698 | 0.68 |
| Decision Tree | 0.680 | 0.66 |
| k-Nearest Neighbors | 0.674 | 0.65 |

Random Forest leads on every class. All four models find the "Good"
class hardest, which tracks with it being the smallest.

## Streamlit app

`Streamlit_Credit_Score_Classification.py` provides a sidebar selector
for seven exploratory plots (age distribution, occupation counts,
correlation heatmap, income by credit score, and others) and a second
selector to train and evaluate any of the four models on demand,
showing accuracy, confusion matrix and classification report.

Run it with:

    streamlit run Streamlit_Credit_Score_Classification.py

## Setup

    pip install -r requirements.txt

## Files

| File | Purpose |
|---|---|
| `Credit_Score_Classification.ipynb` | Full cleaning, EDA and model training |
| `Streamlit_Credit_Score_Classification.py` | Interactive dashboard |
| `requirements.txt` | Dependencies |

## Notes

The Kaggle test set is unlabelled, so all reported metrics come from a
validation split of the training data. Predictions on the test set are
generated but cannot be scored.

---

**Zohreh Taghibakhshi** · [GitHub](https://github.com/zizo155)
