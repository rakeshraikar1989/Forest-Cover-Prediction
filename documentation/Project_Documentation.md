# Project Documentation – Forest Cover Type Prediction

## 1. Introduction
This project develops a machine learning classification solution to predict forest cover types using cartographic and environmental measurements.

The dataset contains 15,120 records, 56 original columns and seven forest cover classes.

## 2. Problem Statement
The objective is to predict the correct forest cover class from geographical and environmental features. This is a seven-class classification problem.

## 3. Dataset Understanding and Quality Checks
The dataset was examined for:
- Number of records and columns
- Data types
- Missing values
- Duplicate records
- Constant columns
- Target class distribution

The dataset contains 15,120 records and 56 columns. All columns are integer-valued. No missing values or duplicate rows were found.

Each of the seven target classes contains 2,160 records. The target is therefore perfectly balanced in this dataset.

Soil_Type7 and Soil_Type15 contain only zero values and were removed because they provide no predictive variation.

## 4. Exploratory Data Analysis
The exploratory analysis includes:
- Target class distribution
- Frequency of wilderness and soil indicator features
- Numerical feature descriptive statistics and distributions
- Skewness analysis
- IQR-based outlier summaries
- Correlation analysis
- Feature relationships with target classes
- Elevation distribution by cover type

The analysis was used to understand the data structure, identify redundant features and examine how environmental measurements vary across cover types.

Potential outliers were retained. Tree-based models can handle different feature scales, and extreme environmental measurements may represent valid observations rather than errors.

## 5. Data Preprocessing
The identifier column Id was excluded because it is not an environmental predictor.

Soil_Type7 and Soil_Type15 were removed because both are constant.

The final modelling dataset contains 52 predictors and the target variable Cover_Type.

Preprocessing decisions:
- Missing-value imputation was unnecessary because no missing values were present.
- Encoding was unnecessary because categorical indicators were already represented as binary columns.
- Feature scaling was not applied because the selected algorithms are tree-based and do not require scaling.
- Stratified sampling was used to preserve class proportions.

The final model is saved as a Scikit-learn pipeline.

## 6. Baseline and Model Comparison
A DummyClassifier was used as a baseline.

The following classifiers were evaluated:
1. Decision Tree
2. Random Forest
3. Extra Trees
4. HistGradientBoosting

RepeatedStratifiedKFold was used with three folds and two repeats.

Models were compared using:
- Accuracy
- Macro F1
- Balanced accuracy
- Mean and standard deviation across validation runs
- Total cross-validation runtime

Extra Trees achieved the highest mean cross-validation macro F1 of 0.8554, with a standard deviation of 0.0038.

Random Forest achieved a mean macro F1 of 0.8536. HistGradientBoosting achieved 0.8454, while Decision Tree achieved 0.7760.

## 7. Hyperparameter Tuning
RandomizedSearchCV was applied to the Extra Trees model.

The search used stratified three-fold cross-validation and macro F1 as the scoring metric. This scoring choice evaluates performance across all classes rather than relying only on overall accuracy.

The selected tuned model was evaluated on the stratified holdout set.

## 8. Final Holdout Evaluation
The tuned Extra Trees model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 87.57% |
| Macro F1 | 87.43% |
| Weighted F1 | 87.43% |
| Balanced Accuracy | 87.57% |

The holdout set contains 3,024 records, with 432 records per class.

Class-level recall:
- Cover Type 1: 80.79%
- Cover Type 2: 71.99%
- Cover Type 3: 83.10%
- Cover Type 4: 97.22%
- Cover Type 5: 96.06%
- Cover Type 6: 87.96%
- Cover Type 7: 95.83%

Cover Type 2 had the lowest recall and deserves further investigation.

## 9. Error Analysis
The classification report and confusion matrix were used to examine model performance by class.

The analysis includes class-level precision, recall and F1 scores, as well as the most frequent class-confusion pairs.

This helps identify classes that the model finds difficult to distinguish.

## 10. Feature Importance
Two feature-importance methods were examined:
- Impurity-based feature importance from Extra Trees
- Permutation importance evaluated on the holdout features using macro F1

Elevation ranked as the most influential feature in both analyses.

Feature importance indicates how the fitted model uses available information. It does not establish that a feature causes a particular forest cover type.

## 11. Saved Outputs
The project generates:
- Executed Jupyter notebook
- Final PDF report
- Exploratory analysis charts
- Cross-validation comparison results
- Hyperparameter tuning results
- Holdout evaluation metrics
- Classification report
- Confusion matrix
- Class error analysis
- Impurity and permutation feature importance
- Saved model pipeline and feature schema

## 12. Limitations
The dataset was used in earlier project iterations. The revised holdout set must therefore not be described as historically untouched.

Cross-validation and holdout results are internal estimates based on this dataset. They do not replace evaluation on an independently collected dataset or geographic region.

The dataset contains 15,120 records and may not represent all forest environments or collection conditions.

Feature importance is not evidence of causation.

## 13. Conclusion
The project developed and evaluated a seven-class forest cover prediction model.

Extra Trees achieved the strongest mean cross-validation macro F1 among the compared default candidates. After hyperparameter tuning, the final model achieved 87.57% holdout accuracy and 87.43% holdout macro F1.

The model performed particularly well for Cover Types 4, 5 and 7. Cover Type 2 had the lowest recall at 71.99%, making it an important area for further analysis.

The results demonstrate useful predictive performance on the available dataset, while independent geographic validation remains necessary before real-world deployment.

## 14. Future Work
- Evaluate the model on independently collected data.
- Investigate confusion between Cover Type 2 and other classes.
- Compare additional classification algorithms.
- Explore justified feature engineering.
- Evaluate model generalisation across geographic regions.
- Consider a deployment interface after further validation.
