---
wave: 1
depends_on: []
files_modified: ["requirements.txt", "src/features/build_features.py", "src/models/train.py", "scripts/test_ml.py"]
autonomous: true
---

# Plan 1: ML Model Training Pipeline

## Objective
Train separate Random Forest models per variable using expanded rolling features, and extract evaluation metrics plus native feature importances.

## Tasks

<task>
<action>
Update `requirements.txt` to append `scikit-learn==1.4.1.post1`.
</action>
<read_first>
`requirements.txt`
</read_first>
<acceptance_criteria>
`cat requirements.txt | grep scikit-learn` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create directory `src/features/` and build `src/features/build_features.py`. Implement `create_time_series_features(df, lags=[1, 3, 7], rolling_windows=[7, 14])`. Add lagged values using `shift()`, moving averages using `rolling(window).mean()`, and percentage rate of change (ROC) using `pct_change(window)` for all numerical columns. Drop rows with resulting NaNs (`dropna()`).
</action>
<read_first>
`src/data/ingest.py`
</read_first>
<acceptance_criteria>
`cat src/features/build_features.py | grep "def create_time_series_features"` returns a match.
`cat src/features/build_features.py | grep "shift"` returns a match.
`cat src/features/build_features.py | grep "rolling"` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create directory `src/models/` and build `src/models/train.py`. Implement `train_models(df, target_cols)`. For each target column in `target_cols`:
1. Use shifted Target (T+1) as the label using `shift(-1)`.
2. Apply temporal train-test split (chronological split, no random shuffles).
3. Train `RandomForestRegressor(random_state=42)`.
4. Import `mean_squared_error`, `mean_absolute_error`, `r2_score` from `sklearn.metrics`. Log RMSE, MAE, R².
5. Extract native `feature_importances_` and build a dataframe of top features mapped to the target.
Return dictionary containing models, aggregated metrics, and importances.
</action>
<read_first>
`src/features/build_features.py`
</read_first>
<acceptance_criteria>
`cat src/models/train.py | grep "RandomForestRegressor"` returns a match.
`cat src/models/train.py | grep "feature_importances_"` returns a match.
`cat src/models/train.py | grep "shift(-1)"` returns a match.
</acceptance_criteria>
</task>

<task>
<action>
Create `scripts/test_ml.py`. Write a script that uses `src.data.ingest` and `src.features.build_features` on `test_mock_data.csv` (using same generator logic as Phase 1). Then it should import `train_models` from `src.models.train` to train the test data and print the resulting dictionary (including R2 and top feature importances).
</action>
<read_first>
`scripts/test_ingest.py`
</read_first>
<acceptance_criteria>
Running `python scripts/test_ml.py` executes successfully with exit code 0 and prints R2 scores.
</acceptance_criteria>
</task>

## Verification
- System splits data temporally without leakage (train/test chronological split implemented in `train.py`).
- Model predicts T+1 with baseline metrics logged (RMSE, MAE, R²).
- Feature importance analysis is performable for each relationship (extracted per target).
- Basic explainability reports can be generated (driven by native RF importances).

## Must Haves
- Separate Random Forest regressors per target variable.
- Features include moving averages and lags.
- Feature importance arrays are generated using Scikit-Learn natively.
