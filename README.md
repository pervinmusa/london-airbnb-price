# London Airbnb Price Prediction

# Machine Learning Model Documentation

## Project Overview

Regression task: predict the **nightly price (£)** of an Airbnb listing in London from its characteristics.

| | |
|---|---|
| **Data source** | [Inside Airbnb](http://insideairbnb.com/get-the-data/), London, scraped June 2026 |
| **Storage** | PostgreSQL (database `airbnb_london`, table `listings`) |
| **Raw size** | 92,638 listings, 90 columns |
| **Modelling dataset** | 61,648 listings (price between £15 and £1,500) |
| **Train / test** | 49,318 / 12,330 (80% / 20%) |
| **Target** | `log1p(price)`; predictions are converted back to £ with `expm1` |
| **Final model** | LightGBM regressor inside a scikit-learn `Pipeline` (33 selected features) |
| **Main metrics** | R² (log scale) and MAE (£) |
| **Deliverables** | Analysis notebook + Streamlit price estimator app |

The project has two parts:

1. **Notebook** (`final_project_airbnb.ipynb`): data loading, cleaning, feature engineering, EDA, model comparison, tuning, feature selection, error analysis, and export of the app model.
2. **Streamlit app** (`app.py`): a host enters 10 facts about a listing and gets a typical nightly price.

### Key Results

| Metric | Value |
|---|---|
| **Test R²** | **0.822** |
| **Test MAE** | **£55.6** (mean test price £238, median £181) |
| Cross-validated R² / MAE (train only) | 0.813 / £56.3 |
| Baseline (always predict the median) | R² ≈ 0, MAE £134.0 |
| **App model (10 inputs)** | **R² 0.754, MAE £65.8** |

**R²** = share of price variation explained (1 = perfect). **MAE** = average error in pounds.

---

## Exploratory Data Analysis

### Data Quality Issues Identified

- **Skewed target:** the raw price has a long right tail; `log1p(price)` is close to bell-shaped.
- **Missing prices:** about 33% of the rows (30,398 of 92,638) have no valid price in the source data.
- **Fully empty columns:** 9 columns contain only missing values.
- **Leakage columns:** columns derived from price (`price_quote_*`, `estimated_revenue_l365d`, `estimated_occupancy_l365d`).
- **Non-informative columns:** identifiers, URLs, scraping metadata, host name, licence.
- **Redundant columns:** `neighbourhood_group_cleansed`, `calendar_updated`, `neighbourhood` (duplicated by `neighbourhood_cleansed`).
- **Highly correlated numeric features:** several night-limit variants, 60/90-day availability, review sub-scores.
- **Near-constant flags:** `host_identity_verified` (98.6% / 1.3%), `has_availability` (99.8% / 0.2%), `host_has_profile_pic` (93.4% / 6.5%).
- **Listings without reviews (~21.8%):** structural missing values in review columns, handled with a `has_reviews` flag.
- **Rare categories:** `property_type` has 83 values with a long tail.

### Price Range and Outliers

The price distribution has a long right tail, so extreme values were examined before modelling.

- An **IQR rule on the log scale** gives bounds of about **£18.6 – £1,547.6**.
- Based on these bounds, the round limits **£15 and £1,500** per night were chosen.
- Listings below £15: **71**. Listings above £1,500: **521**. Together **592 rows (1.0%)** were treated as outliers and removed.
- The filter is applied to the whole dataset before the train/test split, so train and test cover the same price range.

### Key Findings

- **Location, capacity and room type matter most.** Distance to the centre, `accommodates` and `room_type` are the strongest features.
- **Neighbourhood medians differ sharply:** about £85 to £315 across boroughs.
- **Room type medians differ sharply:** from about £41 to about £237.
- **No single feature explains the price.** Performance comes from combining many moderately informative features, which is why tree boosting works well.

---

## Preprocessing Pipeline

### Steps Applied

1. **Duplicate check** on listing `id` (before the ID column is dropped).
2. **Drop non-informative columns:** IDs, URLs, scraping metadata, host name, licence.
3. **Drop leakage columns:** everything derived from price.
4. **Drop fully empty columns** (9).
5. **Price cleaning:** parse to number, remove missing and non-positive prices (92,638 → 62,240 rows).
6. **Merge tenure columns:** years + months merged into `user_years_total` and `host_years_total`.
7. **Remove correlated features:** one column kept from each group with |corr| > 0.9, then > 0.8. The target was not used, so there is no leakage.
8. **Drop near-constant flags.**
9. **Feature engineering** (table below).
10. **Price range filter:** keep £15 ≤ price ≤ £1,500 (61,648 rows) and create `log_price`.
11. **Train / test split:** 80% / 20%, `random_state=42`.
12. **Outlier capping** of numeric features at the 99th or 99.9th percentile, **computed on train only** and applied to train and test.
13. **Model pipeline** (fitted on training folds only):

| Column group | Count | Processing |
|---|---|---|
| Numeric | 26 | median imputation (+ standard scaling for Ridge only) |
| Binary flags | 23 | most-frequent imputation |
| Categorical | 3 | most-frequent imputation, then one-hot encoding with `min_frequency=30` (rare categories grouped as "infrequent") |

### Feature Engineering

| Group | Features created |
|---|---|
| **Location** | `dist_center_km`: haversine distance to Charing Cross (51.5074, −0.1278); `host_in_london` |
| **Reviews** | `days_since_first_review`, `days_since_last_review`, `review_span_days`, `has_reviews` |
| **Bathrooms** | `bath_shared`; missing `bathrooms` filled from `bathrooms_text` ("half-bath" = 0.5) |
| **Host tenure** | `user_years_total`, `host_years_total` |
| **Amenities** | `n_amenities` (count) + 19 hand-picked amenity flags (dishwasher, washer, dryer, TV, elevator, kitchen, dedicated workspace, self check-in, parking, etc.) |

### What We Did NOT Do

- **No IQR-based removal of numeric features' outliers:** it would flag 8–17% of rows and cut off professional hosts; only extreme values were capped.
- **No scaling for tree models:** trees are scale-invariant (scaling is used only for Ridge).
- **No manual grouping of `property_type`:** `OneHotEncoder(min_frequency=30)` does it inside the pipeline.
- **No text features** from `name` / `description`.
- **No tuning of Ridge and Random Forest:** only LightGBM was tuned.

---

## Model Selection & Training

### Validation Strategy

- All decisions (model choice, tuning, number of features) use **3-fold cross-validation on the training set**, with the **same folds for every model**.
- **CV MAE (£) decides**; CV R² is checked next to it.
- The **test set is used only to report** the final results.

### Models Evaluated

| Model | Configuration | Why it is included |
|---|---|---|
| **Dummy** | `strategy="median"` | Lowest bar: every model must beat it |
| **Ridge** | `alpha=1.0` | Simple linear baseline, interpretable coefficients |
| **Random Forest** | 200 trees, `min_samples_leaf=3` | Non-linear baseline, robust and stable |
| **LightGBM** | tuned | Learns non-linear effects and interactions (e.g. location × room type) |

### Results

| Model | Test R² | Test MAE (£) | CV R² | CV MAE (£) |
|---|---|---|---|---|
| Dummy (median) | 0.000 | 134.0 | – | – |
| Ridge | 0.745 | 69.1 | 0.736 | 68.8 |
| Random Forest | 0.797 | 59.0 | 0.782 | 60.5 |
| LightGBM, tuned (52 features) | 0.823 | 55.4 | 0.814 | 56.2 |
| **LightGBM final (33 features)** | **0.822** | **55.6** | 0.813 | 56.3 |

- **LightGBM beats Ridge by about £14 MAE and Random Forest by about £3.5 MAE.**
- Random Forest overfits more (train R² 0.942 vs test 0.797).

### Hyperparameter Tuning

`RandomizedSearchCV` with 20 candidates, 3-fold CV; MAE (£) selects the winner, R² is tracked next to it.

**Best hyperparameters (used for the final model):**

```python
{
    "n_estimators": 1500,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_lambda": 1,
    "random_state": 42,
}
```

A second, more regularised search gave a smaller overfitting gap (0.040) but a slightly worse CV MAE (£57.6), so the first set of parameters was kept.

### Feature Selection

Permutation importance was computed on a validation part of the training data. Candidate feature sets were compared by CV and the **one-standard-error rule** picked the smallest set within one standard error of the best CV MAE.

| Feature set | Features | CV R² | CV MAE (£) |
|---|---|---|---|
| All | 52 | 0.814 | 56.18 |
| threshold 0.0005 | 41 | 0.814 | 56.24 |
| **threshold 0.001 (chosen)** | **33** | **0.813** | **56.34** |
| threshold 0.003 | 25 | 0.811 | 56.71 |
| threshold 0.005 | 20 | 0.805 | 57.45 |

Going from 52 to 33 features costs only about **£0.15 of CV MAE**, which is within noise, and gives a simpler model.

### Top Features (permutation importance)

| Feature | Importance (drop in R²) |
|---|---|
| `dist_center_km` | 0.136 |
| `accommodates` | 0.113 |
| `room_type` | 0.111 |
| `minimum_nights` | 0.059 |
| `availability_30` | 0.045 |
| `bedrooms` | 0.044 |

---

## Final Model Performance

| | R² | MAE (£) |
|---|---|---|
| Train | 0.892 | 44.0 |
| **Test** | **0.822** | **55.6** |

- **R² gap (train − test): 0.070**, in the normal range (< 0.10).
- A **validation curve** (`num_leaves`) and a **learning curve** (training size) are included in the notebook to check the bias–variance balance.

### Error Analysis

| True price band | Mean abs. error (£) | Mean % error | n |
|---|---|---|---|
| £0 – 75 | 20.0 | 40.4% | 1,826 |
| £75 – 150 | 25.6 | 23.2% | 3,264 |
| £150 – 250 | 40.7 | 20.6% | 3,119 |
| £250 – 400 | 61.4 | 19.6% | 2,311 |
| £400 – 700 | 110.6 | 21.7% | 1,293 |
| £700 – 1,500 | 296.8 | 30.7% | 517 |

- The model is **most reliable between £75 and £400** (about 20% error).
- The **largest mistakes are expensive listings predicted far too low**, mostly in Westminster and Kensington and Chelsea.
- Cheap listings have a high % error because a small £ error is a large share of a small price.

### Host-Level Check

`host_id` is **not** a feature, but the same host can appear in both train and test. To measure the effect, a random split was compared with a split grouped by host:

| Split | Test R² | Test MAE (£) | Hosts in both train and test |
|---|---|---|---|
| Random (used in the notebook) | 0.822 | 55.6 | 3,801 |
| Grouped by host | 0.793 | 60.7 | 0 |

The difference is small, so the model is stable. For a **brand-new host**, the second row is the more realistic estimate.

---

## App Model

The full model uses features a host may not know (reviews, availability, etc.). The app therefore uses a **separate model trained only on 10 host-known features**:

`neighbourhood_cleansed`, `room_type`, `property_type`, `accommodates`, `bedrooms`, `bathrooms`, `bath_shared`, `minimum_nights`, `n_amenities`, `dist_center_km`

| Approach | Test R² | Test MAE (£) |
|---|---|---|
| **A. Separate model trained on the 10 app features (used)** | **0.754** | **£65.8** |
| B. Full 33-feature model, missing fields set to NaN | 0.702 | £74.5 |

---

## Streamlit App

### What it does

A web page with two modes:

1. **Host mode:** "What price should I expect for my listing?"
2. **Check mode:** "Is the price of this listing in line with similar listings?"

### Inputs

Neighbourhood (33 boroughs), room type, property type (15 most common + "Other"), shared bathroom, guests (1–10), bedrooms (1–5), bathrooms (0.5–3.5), minimum nights (1–30), number of amenities (4–65), optional exact coordinates, and (check mode only) the listed price per night. Ranges come from `ranges.json`; built-in defaults are used if it is missing.

### Outputs

- **Typical price per night** (£) and the **distance to central London** used.
- **Likely price range** (only if the optional quantile models are present).
- **Check mode:** the listed price as a % above or below the estimate (plus a label *below / within / above the typical range* when a range exists).
- An **"About this estimate"** box with data source, valid price range and test error.

By default the distance is measured from the **median point of the chosen neighbourhood**; if exact coordinates are entered, those are used (more accurate, because the model was trained with exact coordinates).

### App Files

| File | Required | Purpose |
|---|---|---|
| `app.py` | yes | The Streamlit app |
| `app_model.joblib` | **yes** | Trained 10-feature pipeline |
| `neighbourhood_centers.json` | **yes** | Neighbourhood → median latitude / longitude |
| `options.json` | **yes** | Dropdown values |
| `ranges.json` | optional | Min / max for numeric inputs (defaults used if missing) |
| `app_info.json` | optional | Text shown in "About" |
| `app_model_q10.joblib`, `app_model_q90.joblib` | optional | Quantile models for the likely price range |

The app only **reads** files produced by the notebook. After re-running the notebook, replace these files; `app.py` does not change.

---

## Project Structure

```
project/
├── app.py                         # Streamlit web interface
├── requirements.txt               # Python dependencies (app)
├── app_model.joblib               # Trained app model (10 features)
├── neighbourhood_centers.json     # Neighbourhood -> median lat/lon
├── options.json                   # Dropdown options
├── final_project_airbnb.ipynb     # Full analysis and modelling notebook
└── README.md
```

---

## Setup & Installation

### Prerequisites

- **Python 3** (a recent version; the pinned packages need a modern Python)
- **Environment:** virtualenv or Conda recommended
- **PostgreSQL** (only needed to re-run the notebook from raw data)

### Installation Steps

```bash
git clone <https://github.com/pervinmusa/london-airbnb-price>
cd <london-airbnb-price>
python -m venv .venv
.venv\Scripts\activate             # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

`requirements.txt` contains what the **app** needs:

```
joblib==1.6.0
lightgbm==4.7.0
numpy==2.5.4
pandas==3.0.6
scikit-learn==1.9.1
streamlit
```

To run the **notebook**, also install:

```bash
pip install sqlalchemy psycopg2-binary matplotlib seaborn scipy jupyter
```

> **Version note:** `app_model.joblib` is a pickled scikit-learn / LightGBM pipeline and must be loaded with the **same versions** it was trained with.

---

## Training the Model

### Data Preparation

1. Download the London `listings.csv.gz` from [Inside Airbnb](http://insideairbnb.com/get-the-data/) (June 2026 snapshot).
2. Load it into PostgreSQL: database `airbnb_london`, table `listings`.
3. Provide the database password. The notebook asks for it with hidden input and never stores it; you can also set it in advance:

```powershell
$env:PG_PASSWORD="your_password"
```

If your connection differs (user, host, port), edit the `create_engine(...)` call in the first code cell (default: user `postgres`, `localhost:5432`).

### Run Training

1. Start Jupyter: `jupyter notebook`
2. Open `final_project_airbnb.ipynb` and run all cells from top to bottom.
3. The last cells save `app_model.joblib`, `neighbourhood_centers.json` and `options.json`.

All random seeds are fixed (`random_state=42`), so results are reproducible for the same data snapshot and library versions.

---

## Making Predictions

### Method 1: Streamlit Web Interface

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Choose a mode, fill in the listing details and press **Estimate**.

### Method 2: Python

```python
import joblib, numpy as np, pandas as pd

model = joblib.load("app_model.joblib")
row = pd.DataFrame([{
    "neighbourhood_cleansed": "Camden", "room_type": "Entire home/apt",
    "property_type": "Entire rental unit", "accommodates": 2.0, "bedrooms": 1.0,
    "bathrooms": 1.0, "bath_shared": 0, "minimum_nights": 2.0,
    "n_amenities": 25.0, "dist_center_km": 3.0,
}])
print(f"£{np.expm1(model.predict(row))[0]:,.0f} per night")   # the model predicts log1p(price)
```

Columns must have **exactly these names and this order**.

---

## Model Monitoring Recommendations

Recommendations only (no drift analysis was run in this project): track whether the input mix (neighbourhood, room type, guests, amenities) and the predicted prices drift away from the training data (e.g. PSI > 0.25 = major shift), track MAE when real prices are available, and retrain with a fresh Inside Airbnb snapshot regularly.

---

## Limitations

These come from the data itself:

1. **Missing prices:** about 33% of the listings (30,398 of 92,638) have no price in the source data, so the model learned only from listings that have one.
2. **Few expensive listings:** listings above £700 are rare and very different from each other, so the model underestimates them (mean error ≈ £297).
3. **Very rare room types:** the test set has only 17 hotel rooms and 31 shared rooms, so results for these are not reliable.
4. **Single snapshot:** all data comes from one scrape (June 2026), so seasonality and price trends are not covered.
5. **Asking prices:** the data contains prices hosts currently ask, not confirmed booking prices. The model reflects the market level, not the optimal price.
6. **Availability:** `availability_30` is an important feature but depends on the scraping date and is not known to a host in the app, so the app model does not use it.

---

## Key Decisions Summary

| Decision | Rationale |
|---|---|
| Predict `log1p(price)` | Price is strongly right-skewed; log makes errors relative and stabilises training |
| Keep £15 – £1,500 only | Close to the IQR bounds on the log scale (£18.6 – £1,547.6); extreme prices treated as outliers (1.0% of rows) |
| Drop price-derived columns | Avoid data leakage |
| Drop correlated columns (\|corr\| > 0.8) | Reduce redundancy; target not used, so no leakage |
| Caps computed on train only | Limit extreme values without using the test set |
| Imputation and encoding inside the pipeline | Fitted on training folds only, so CV is honest |
| `OneHotEncoder(min_frequency=30)` | Handles the long tail of `property_type` automatically |
| 3-fold CV, CV MAE decides | Test set stays untouched for the final report |
| LightGBM over Ridge / Random Forest | Best CV and test MAE with a normal overfitting gap |
| 33 features (1-SE rule) | Simpler model with no meaningful loss (≈ £0.15 MAE) |
| Host-level check | Measures how much the same host in train and test affects the result |
| Separate 10-feature app model | Better than setting missing fields to NaN (R² 0.754 vs 0.702) |

---

## Troubleshooting

**Issue:** `FileNotFoundError: app_model.joblib`
**Solution:** Place `app_model.joblib` and the two JSON files in the same folder as `app.py`, or re-run the notebook to create them.

**Issue:** Model fails to load or gives odd results
**Solution:** Install the exact versions from `requirements.txt` (scikit-learn 1.9.1, LightGBM 4.7.0).

---

## Important Notes

- **Model and JSON files** must sit next to `app.py`.
- **The notebook reads data from PostgreSQL**, not from a CSV file.
- **The password is never stored** in the repository; it is requested at runtime or read from the `PG_PASSWORD` environment variable.
- **Data license:** [Inside Airbnb](http://insideairbnb.com/) data is available under a Creative Commons Attribution 4.0 license. This is an educational project and is not affiliated with Airbnb.

---

## Contact & Questions

**Author:** Pərvin Musayeva
**Date:** 20.11.2026
**Data snapshot:** Inside Airbnb London, June 2026
**Framework:** LightGBM 4.7.0, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4
**Environment:** Windows 11
