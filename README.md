\# Forest Cover Type Prediction



\## Project Overview

This project uses machine learning to predict forest cover types from cartographic and environmental features. The objective is to classify forest areas into one of seven cover types using the provided dataset.



\## Project Details

\- \*\*Project Code:\*\* PRCP-1005

\- \*\*Project:\*\* Forest Cover Prediction

\- \*\*Problem Type:\*\* Multiclass Classification

\- \*\*Dataset Records:\*\* 15,120

\- \*\*Input Features:\*\* 52

\- \*\*Target Classes:\*\* 7



\## Machine Learning Models

The following models were evaluated:

1\. Decision Tree

2\. Random Forest

3\. Extra Trees

4\. Hist Gradient Boosting



Extra Trees achieved the best initial comparison accuracy of 85.65%.



Hyperparameter tuning was performed using RandomizedSearchCV with 3-fold Stratified Cross-Validation.



\## Final Tuned Model

\- \*\*Algorithm:\*\* Extra Trees Classifier

\- \*\*Cross-Validation Accuracy:\*\* 83.78%

\- \*\*Final Test Accuracy:\*\* 84.85%



\## Important Features

The top features included:

\- Elevation

\- Wilderness\_Area4

\- Horizontal\_Distance\_To\_Roadways

\- Horizontal\_Distance\_To\_Fire\_Points

\- Horizontal\_Distance\_To\_Hydrology



\## Project Structure

```text

Forest-Cover-Prediction/

├── data/

├── documentation/

├── models/

├── notebooks/

├── reports/

│   └── charts/

├── .gitignore

├── README.md

└── requirements.txt

```



\## Technologies Used

\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Scikit-learn

\- Jupyter Notebook

\- Joblib

\- Git



\## How to Run the Project



1\. Clone or download this repository.

2\. Create and activate a Python virtual environment.

3\. Install dependencies:



```bash

pip install -r requirements.txt

```



4\. Place the dataset file `train.csv` inside the `data/` folder.

5\. Open the notebook:



```bash

jupyter notebook

```



6\. Open `notebooks/PRCP\_1005\_Forest\_Cover\_Prediction.ipynb` and run the cells.



\## Model File

The trained model is saved at:



`models/extra\_trees\_forest\_cover.joblib`



\## Evaluation Reports

The `reports/` folder contains:

\- Model comparison

\- Model metrics

\- Classification report

\- Feature importance

\- Confusion matrix chart

\- Feature importance chart



\## Notes

The dataset used in this project contains 15,120 records. The initial model comparison used the test split for exploratory comparison, and the tuned model was subsequently evaluated on that same test split. Therefore, the final test result should be interpreted with this model-selection limitation in mind.



The dataset CSV files and trained model are excluded from Git tracking through `.gitignore`.



\## Project Status

Core model development, evaluation, tuning, and artifact generation are complete. Documentation and repository preparation are in progress.

