# ML-Regression-and-Hypertuning

# ML Assignment 4 — Regression Evaluation Metrics (California Housing)

## Objective
Evaluate multiple regression algorithms on the California Housing dataset and compare their performance using RMSE, MAE, and R², with 5-fold cross-validation and hyperparameter tuning.

## Dataset
- `sklearn.datasets.fetch_california_housing(as_frame=True)`
- Features: housing + geo/socioeconomic variables; Target: median house value.

## Methods
- **Models**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, SVR (RBF).
- **Pipelines**: Scaling applied for SVR; tree/OLS pipelines left unscaled.
- **Evaluation**: 5-fold CV on train set with RMSE, MAE, R²; final test metrics reported.
- **Tuning**: `GridSearchCV` for top models (typically Gradient Boosting & Random Forest).

## How to Run
1. Install requirements (scikit-learn, numpy, pandas).
2. Open the notebook and run cells in order.
3. The notebook will:
   - Load & split data
   - Run 5-fold CV for all models
   - Tune top models
   - Report final test set metrics and the best model

## Results (Typical)
- **Best**: Gradient Boosting Regressor (lowest RMSE / highest R² after tuning)
- **Runner-up**: Random Forest Regressor
- **Baseline/Worst**: Linear Regression or Decision Tree (under/overfit respectively)

## Files
- `notebook.ipynb` — full code and analysis
- 
