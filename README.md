# Forest Cover Type Prediction

## Project Overview
This project uses machine learning classification to predict forest cover types from cartographic and environmental features.

## Dataset
- Records: 15,120
- Original columns: 56
- Target classes: 7
- Records per class: 2,160
- Missing values: 0
- Duplicate rows: 0
- Constant columns: Soil_Type7 and Soil_Type15
- Predictors retained: 52

## Data Preparation
The identifier column was excluded because it does not represent an environmental predictor. Two constant soil indicator columns were removed because they contain no variation.

No imputation was required because the dataset had no missing values. Soil and wilderness indicators were already encoded as binary features. Scaling was not applied because the selected tree-based algorithms do not require feature scaling.

## Exploratory Data Analysis
The analysis includes:
- Target class distribution
- Binary indicator frequency analysis
- Numerical feature distributions and skewness
- Outlier summaries using the IQR method
- Feature correlation analysis
- Feature relationships with forest cover classes
- Elevation comparisons across classes

Potential outliers were retained because extreme environmental measurements may represent valid geographical conditions.

## Models and Validation
Four classifiers were compared:
- Decision Tree
- Random Forest
- Extra Trees
- HistGradientBoosting

Repeated stratified 3-fold cross-validation, repeated twice, was used to compare accuracy, macro F1 and balanced accuracy.

Extra Trees achieved the highest mean cross-validation macro F1.

## Final Model Results
The tuned Extra Trees model achieved the following holdout results:

| Metric | Result |
|---|---:|
| Accuracy | 87.57% |
| Macro F1 | 87.43% |
| Weighted F1 | 87.43% |
| Balanced Accuracy | 87.57% |

The lowest class recall was for Cover Type 2 at 71.99%.

## Feature Interpretation
Both impurity-based and permutation feature importance were examined. Elevation was the most influential feature in both analyses. Feature importance describes model behaviour and does not establish causation.

## Project Outputs
- Jupyter notebook: `notebooks/PRCP_1005_Forest_Cover_Prediction.ipynb`
- Final PDF report: `reports/Forest_Cover_Prediction_Final_Report.pdf`
- Analysis charts: `reports/charts/`
- Evaluation results: `reports/`
- Report generator: `generate_report.py`

The trained model and feature schema are saved under `models/` but are excluded from Git because of their size.

## Limitations
The dataset was used in earlier project iterations. Therefore, the revised holdout is not historically untouched. Cross-validation and holdout results are internal estimates, not independent geographic validation. Performance may differ on data from other regions or collection conditions.

## Future Work
- Validate using an independently collected dataset.
- Investigate frequent class-confusion pairs.
- Evaluate additional algorithms and justified feature engineering.
- Test generalisation across different geographic regions.

## Technology
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook and ReportLab.
