# Challenges Faced and Decisions – Forest Cover Type Prediction

## 1. Dataset Quality
The initial dataset audit checked missing values, duplicate records, data types and constant columns.

**Finding:** No missing values or duplicate rows were present. Soil_Type7 and Soil_Type15 contained only zeros.

**Decision:** The two constant columns were removed because they provide no useful variation for classification.

## 2. Target Class Distribution
All seven cover types contained exactly 2,160 records.

**Decision:** Stratified splitting and validation were used to preserve class proportions. The dataset was not treated as imbalanced.

## 3. Feature Preparation
The dataset included an identifier, binary soil and wilderness indicators, and numerical environmental measurements.

**Decisions:**
- Excluded Id because it is an identifier rather than an environmental measurement.
- Retained the existing binary indicators without additional encoding.
- Did not impute because there were no missing values.
- Did not scale because the evaluated algorithms were tree-based.
- Retained potential IQR outliers because they may represent valid environmental conditions.

## 4. Model Selection
Several tree-based classifiers were compared using repeated stratified cross-validation.

**Finding:** Extra Trees achieved the highest mean cross-validation macro F1 of 0.8554.

**Decision:** Extra Trees was selected for hyperparameter tuning.

## 5. Hyperparameter Tuning
The tuning process used RandomizedSearchCV with stratified three-fold cross-validation and macro F1 scoring.

**Reason:** Macro F1 gives equal importance to each class and provides a more suitable comparison than accuracy alone.

## 6. Class-Level Performance
The final holdout evaluation achieved 87.57% accuracy and 87.43% macro F1.

Cover Type 2 had the lowest recall at 71.99%, while Cover Types 4 and 7 achieved recall above 95%.

**Decision:** Class-level recall and confusion analysis were included so that overall accuracy would not hide weaker class performance.

## 7. Feature Interpretation
Impurity-based feature importance and permutation importance were both examined.

**Finding:** Elevation ranked highest in both analyses.

**Caution:** Feature importance describes the model's behaviour and should not be interpreted as proof of causation.

## 8. Validation Limitation
The dataset had been used in earlier project iterations. The revised holdout therefore cannot be described as completely untouched throughout the project history.

**Decision:** This limitation is disclosed in the notebook and final report. The results are internal estimates and require further validation using independently collected data.

## 9. Improvements Made
The revised project includes:
- More detailed exploratory analysis
- Explicit preprocessing justifications
- A baseline classifier
- Repeated stratified cross-validation
- Multiple evaluation metrics
- Macro F1-based hyperparameter tuning
- Class-level error analysis
- Permutation feature importance
- A saved model pipeline
- Expanded project documentation and a generated PDF report

## 10. Further Improvements
Future work should focus on independent geographic validation, investigation of Cover Type 2 errors, evaluation of additional models and assessment of performance on new data.
