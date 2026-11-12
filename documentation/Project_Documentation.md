\# Project Documentation – Forest Cover Type Prediction



\## 1. Introduction

The purpose of this project is to build a machine learning classification model that predicts forest cover types using environmental and cartographic information.



The project uses a dataset containing 15,120 records and seven forest cover classes.



\## 2. Problem Statement

Forest cover classification helps understand the distribution of different forest types based on geographical and environmental conditions.



This project develops a machine learning solution to predict the correct forest cover class from the available input features.



\## 3. Dataset Description

\- Total records: 15,120

\- Total original columns: 56

\- Input features used: 52

\- Target variable: Cover\_Type

\- Number of classes: 7

\- Missing values: 0

\- Duplicate records: 0



The `Id` column and two constant columns (`Soil\_Type7` and `Soil\_Type15`) were excluded from model training.



\## 4. Data Preparation

The following activities were completed:

\- Loaded the dataset using Pandas.

\- Checked missing values and duplicate records.

\- Examined target class distribution.

\- Identified constant columns.

\- Validated Wilderness Area and Soil Type indicator columns.

\- Separated input features and target variable.

\- Split the dataset into training and testing sets using an 80:20 ratio.

\- Used stratification to preserve class distribution.



Training records: 12,096  

Testing records: 3,024



\## 5. Exploratory Data Analysis

The target variable contains seven classes, with 2,160 records in each class.



A class distribution chart was generated to examine the balance of the dataset.



\## 6. Models Evaluated



| Model | Accuracy |

|---|---:|

| Extra Trees | 85.65% |

| Random Forest | 85.42% |

| Hist Gradient Boosting | 85.22% |

| Decision Tree | 79.17% |



Extra Trees achieved the highest initial comparison accuracy.



\## 7. Hyperparameter Tuning

RandomizedSearchCV was used to tune the Extra Trees classifier.



Three-fold Stratified Cross-Validation was applied.



Selected parameters:

\- Number of estimators: 100

\- Maximum depth: 30

\- Minimum samples split: 5

\- Minimum samples leaf: 1

\- Maximum features: sqrt



Best cross-validation accuracy: 83.78%.



\## 8. Final Model Evaluation

The tuned Extra Trees model achieved a test accuracy of 84.85%.



The classification report showed:

\- Macro F1-score: approximately 0.85

\- Weighted F1-score: approximately 0.85



Class 7 achieved the highest F1-score of 0.96, while Class 2 was the most challenging, with an F1-score of 0.70.



A confusion matrix was generated to examine classification performance across all seven classes.



\## 9. Feature Importance

The most influential features were:

1\. Elevation

2\. Wilderness\_Area4

3\. Horizontal\_Distance\_To\_Roadways

4\. Horizontal\_Distance\_To\_Fire\_Points

5\. Horizontal\_Distance\_To\_Hydrology



Elevation had the highest feature importance score.



\## 10. Saved Project Artifacts

\- Trained model: `models/extra\_trees\_forest\_cover.joblib`

\- Model comparison: `reports/model\_comparison.csv`

\- Model metrics: `reports/model\_metrics.csv`

\- Classification report: `reports/classification\_report.csv`

\- Feature importance: `reports/feature\_importance.csv`

\- Confusion matrix: `reports/charts/confusion\_matrix.png`

\- Feature importance chart: `reports/charts/feature\_importance.png`



\## 11. Conclusion

The project successfully developed a seven-class forest cover classification model using the Extra Trees algorithm.



The tuned model achieved 84.85% test accuracy. Elevation was identified as the most influential feature.



The results demonstrate how machine learning can be applied to environmental and geographical classification problems.



\## 12. Evaluation Limitation

The initial model comparison used the test split to compare algorithms. The tuned model was subsequently evaluated on the same test split. Therefore, the final test accuracy is not a completely independent estimate of generalisation performance. A future evaluation should reserve a fresh untouched test set or use nested cross-validation.



\## 13. Future Improvements

\- Evaluate additional classification algorithms.

\- Perform more extensive hyperparameter tuning.

\- Use a fully independent test dataset.

\- Build a simple web application for predictions.

\- Explore deployment through a REST API.

