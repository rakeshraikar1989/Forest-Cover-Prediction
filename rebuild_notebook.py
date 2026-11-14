from pathlib import Path
import nbformat

root = Path.cwd()
notebook_path = root / "notebooks" / "PRCP_1005_Forest_Cover_Prediction.ipynb"
notebook_path.parent.mkdir(parents=True, exist_ok=True)

cells = []

def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbformat.v4.new_code_cell(text))

md("""# Forest Cover Type Prediction
## PRCP-1005 | Machine Learning Classification

**Objective:** Predict one of seven forest cover types using cartographic and environmental variables.

This notebook provides a reproducible workflow covering data validation, exploratory data analysis, preprocessing justification, repeated cross-validation, hyperparameter tuning, final evaluation, error analysis and model interpretation.

**Important validation note:** This dataset was used in earlier project iterations. The revised holdout and cross-validation results must therefore not be described as a completely independent historical evaluation.
""")

md("## 1. Imports and project configuration")

code("""from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

warnings.filterwarnings("default")
RANDOM_STATE = 42

ROOT = Path.cwd()
if not (ROOT / "data" / "train.csv").exists() and (ROOT.parent / "data" / "train.csv").exists():
    ROOT = ROOT.parent
DATA_PATH = ROOT / "data" / "train.csv"
REPORTS = ROOT / "reports"
CHARTS = REPORTS / "charts"
MODELS = ROOT / "models"

for folder in [REPORTS, CHARTS, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 70)
pd.set_option("display.width", 140)

print("Project directory:", ROOT)
print("Dataset path:", DATA_PATH)
print("Dataset exists:", DATA_PATH.exists())
""")

md("## 2. Load and validate the dataset")

code("""if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
display(df.head())
display(df.dtypes.value_counts().rename("Column count").to_frame())
""")

code("""target_col = "Cover_Type"
id_col = "Id"

required_columns = {target_col, id_col}
missing_required = required_columns - set(df.columns)
if missing_required:
    raise ValueError(f"Required columns missing: {missing_required}")

print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Duplicate identifiers:", int(df[id_col].duplicated().sum()))
print("Target classes:", sorted(df[target_col].unique()))
print("\\nTarget counts:")
display(df[target_col].value_counts().sort_index().to_frame("Records"))
""")

md("## 3. Data quality and structural checks")

code("""missing_summary = df.isna().sum().sort_values(ascending=False)
constant_columns = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

audit = pd.DataFrame({
    "Check": [
        "Rows", "Columns", "Missing cells", "Duplicate rows",
        "Duplicate Id values", "Constant columns"
    ],
    "Result": [
        len(df), df.shape[1], int(df.isna().sum().sum()),
        int(df.duplicated().sum()), int(df[id_col].duplicated().sum()),
        ", ".join(constant_columns) if constant_columns else "None"
    ]
})
display(audit)

print("Constant column values:")
for col in constant_columns:
    print(f"{col}: {df[col].iloc[0]}")
""")

md("""### Data quality interpretation

Missing values are checked before modelling. Imputation is not required when the dataset contains no missing observations. Duplicate records and identifier uniqueness are checked separately.

The identifier is excluded because it is a record reference rather than an environmental predictor. Constant predictors contain no variation and cannot help distinguish target classes, so they are excluded from model training. These decisions are based on observed dataset properties, not assumptions.
""")

code("""# Identify predictor groups from the dataset schema.
continuous_candidates = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]
continuous_cols = [c for c in continuous_candidates if c in df.columns]
wilderness_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]

print("Continuous environmental features:", len(continuous_cols))
print("Wilderness indicator features:", len(wilderness_cols))
print("Soil indicator features:", len(soil_cols))
""")

md("## 4. Target distribution")

code("""target_counts = df[target_col].value_counts().sort_index()
target_percent = (target_counts / len(df) * 100).round(2)

target_summary = pd.DataFrame({
    "Records": target_counts,
    "Percentage": target_percent
})
display(target_summary)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=target_counts.index.astype(str), y=target_counts.values, ax=ax)
ax.set(title="Distribution of Forest Cover Classes", xlabel="Cover Type", ylabel="Number of records")
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.savefig(CHARTS / "01_target_distribution.png", dpi=180, bbox_inches="tight")
plt.show()
""")

md("""### Target distribution interpretation

The supplied dataset contains 15,120 records, with 2,160 records in each of the seven classes. The target is therefore exactly balanced in this dataset. Accuracy is useful, but macro F1 and balanced accuracy are also reported to show performance across all classes.
""")

md("## 5. Wilderness and soil indicator analysis")

code("""indicator_cols = wilderness_cols + soil_cols

if indicator_cols:
    invalid_binary = {
        col: sorted(set(df[col].dropna().unique()) - {0, 1})
        for col in indicator_cols
    }
    invalid_binary = {k: v for k, v in invalid_binary.items() if v}

    print("Indicator values other than 0/1:", invalid_binary or "None")

    if wilderness_cols:
        wilderness_row_sums = df[wilderness_cols].sum(axis=1)
        print("Wilderness indicators: row-sum counts")
        display(wilderness_row_sums.value_counts().sort_index().to_frame("Rows"))

    if soil_cols:
        soil_row_sums = df[soil_cols].sum(axis=1)
        print("Soil indicators: row-sum counts")
        display(soil_row_sums.value_counts().sort_index().to_frame("Rows"))

    prevalence = df[indicator_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(x=prevalence.values, y=prevalence.index, ax=ax)
    ax.set(title="Frequency of Wilderness and Soil Indicators", xlabel="Number of active records", ylabel="Indicator")
    plt.tight_layout()
    plt.savefig(CHARTS / "02_indicator_frequency.png", dpi=180, bbox_inches="tight")
    plt.show()
""")

md("""### Indicator interpretation and encoding decision

Wilderness and soil categories are already represented as binary indicator columns. They do not require additional one-hot encoding. The row-sum checks verify whether each observation has exactly one active wilderness category and one active soil category.

Tree-based algorithms can work directly with these numeric indicator columns. No scaling is required for the tree-based candidate models used here.
""")

md("## 6. Numerical feature distributions and descriptive statistics")

code("""numeric_summary = df[continuous_cols].describe().T
numeric_summary["skewness"] = df[continuous_cols].skew()
display(numeric_summary.round(3))

n = len(continuous_cols)
cols = 2
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.4))
axes = np.array(axes).reshape(-1)

for i, col in enumerate(continuous_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f"Distribution: {col}")
    axes[i].set_xlabel(col)

for j in range(n, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(CHARTS / "03_numeric_feature_distributions.png", dpi=180, bbox_inches="tight")
plt.show()
""")

md("## 7. Outlier assessment")

code("""outlier_rows = []
for col in continuous_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df[col] < lower) | (df[col] > upper)).sum())
    outlier_rows.append({
        "Feature": col, "Q1": q1, "Q3": q3,
        "IQR": iqr, "Lower fence": lower,
        "Upper fence": upper, "Potential outliers": count,
        "Outlier percentage": count / len(df) * 100
    })

outlier_summary = pd.DataFrame(outlier_rows).sort_values(
    "Outlier percentage", ascending=False
)
display(outlier_summary.round(2))
outlier_summary.to_csv(REPORTS / "outlier_summary.csv", index=False)
""")

md("""### Outlier treatment decision

The IQR rule is used to flag unusual numerical observations, not automatically delete them. Environmental measurements can legitimately contain extreme values, and tree-based models are generally less sensitive to feature scaling and extreme magnitudes than distance-based models. The observations are retained unless a separate data-quality investigation establishes that a value is erroneous.
""")

md("## 8. Correlation analysis")

code("""corr = df[continuous_cols].corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("Correlation Between Continuous Environmental Features")
plt.tight_layout()
plt.savefig(CHARTS / "04_continuous_feature_correlation.png", dpi=180, bbox_inches="tight")
plt.show()

pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
correlation_pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index)
display(correlation_pairs.head(10).rename("Correlation").to_frame())
display(correlation_pairs.tail(10).rename("Correlation").to_frame())
""")

md("""### Correlation interpretation

Correlation measures linear association between numerical features; it does not demonstrate causation. Strongly correlated predictors may contain overlapping information. Tree ensembles can still use correlated variables, but impurity-based importance may distribute importance across related predictors.
""")

md("## 9. Feature relationships with forest cover type")

code("""class_medians = df.groupby(target_col)[continuous_cols].median()
class_z = (class_medians - class_medians.mean()) / class_medians.std(ddof=0)

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(class_z, cmap="vlag", center=0, annot=True, fmt=".1f", ax=ax)
ax.set(title="Standardised Median Environmental Measurements by Cover Type",
       xlabel="Environmental feature", ylabel="Cover type")
plt.tight_layout()
plt.savefig(CHARTS / "05_feature_target_median_heatmap.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x=target_col, y="Elevation", ax=ax)
ax.set_title("Elevation Distribution by Forest Cover Type")
plt.tight_layout()
plt.savefig(CHARTS / "06_elevation_by_cover_type.png", dpi=180, bbox_inches="tight")
plt.show()
""")

md("""### Feature-target interpretation

The class-level median heatmap compares environmental profiles across cover types. The elevation boxplot gives a more detailed view of how elevation distributions vary between classes. Differences in medians and spreads indicate potentially useful predictive information, while overlap between distributions helps explain why some classes may be confused.
""")

md("## 10. Preprocessing decisions and feature preparation")

code("""# Remove identifier, target and constant columns from the predictors.
excluded_columns = [id_col, target_col] + constant_columns
X = df.drop(columns=excluded_columns, errors="ignore").copy()
y = df[target_col].copy()

# Confirm all remaining features are numeric.
non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()
if non_numeric:
    raise TypeError(f"Non-numeric predictors require review: {non_numeric}")

print("Predictor matrix:", X.shape)
print("Target vector:", y.shape)
print("Excluded columns:", excluded_columns)
print("Remaining missing values:", int(X.isna().sum().sum()))
print("Remaining constant predictors:", [c for c in X.columns if X[c].nunique() <= 1])
""")

md("""### Preprocessing rationale

- **Identifier:** `Id` is excluded because it identifies records rather than describing forest conditions.
- **Constant columns:** columns with no variation are excluded because they cannot separate classes.
- **Missing values:** no imputation is applied because the supplied dataset has no missing values.
- **Categorical encoding:** soil and wilderness categories are already one-hot/binary encoded.
- **Scaling:** not applied because the candidate algorithms are tree-based and do not require standardised feature magnitudes.
- **Pipeline:** each estimator is wrapped in a scikit-learn Pipeline to keep the model workflow consistent and reproducible.

The same predictor columns are used throughout cross-validation, tuning and final evaluation.
""")

md("## 11. Baseline and repeated stratified cross-validation")

code("""baseline = DummyClassifier(strategy="most_frequent")
baseline_scores = cross_validate(
    baseline, X, y,
    cv=RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=RANDOM_STATE),
    scoring={
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy"
    },
    n_jobs=-1,
    return_train_score=False
)

baseline_summary = {
    metric: (np.mean(baseline_scores[f"test_{metric}"]),
             np.std(baseline_scores[f"test_{metric}"], ddof=1))
    for metric in ["accuracy", "macro_f1", "balanced_accuracy"]
}
print("Dummy baseline mean and standard deviation:")
display(pd.DataFrame(baseline_summary, index=["Mean", "Standard deviation"]).T)
""")

code("""candidate_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, random_state=RANDOM_STATE, n_jobs=1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=150, random_state=RANDOM_STATE, n_jobs=1
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=100, random_state=RANDOM_STATE
    )
}

cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=RANDOM_STATE)
scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy"
}

cv_rows = []
for model_name, estimator in candidate_models.items():
    print(f"Evaluating: {model_name}")
    pipeline = Pipeline([("model", clone(estimator))])
    start = time.perf_counter()
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring,
        n_jobs=-1, return_train_score=False
    )
    elapsed = time.perf_counter() - start

    row = {"Model": model_name, "Total CV runtime (seconds)": elapsed}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric} mean"] = values.mean()
        row[f"{metric} std"] = values.std(ddof=1)
    cv_rows.append(row)

cv_results = pd.DataFrame(cv_rows).sort_values("macro_f1 mean", ascending=False)
display(cv_results.round(4))
cv_results.to_csv(REPORTS / "cross_validation_model_comparison.csv", index=False)
""")

md("""### Cross-validation interpretation

Repeated stratified three-fold cross-validation with two repeats produces six validation scores per candidate model. Mean scores describe average performance, while standard deviations show variation across folds. Macro F1 gives equal weight to each class. Balanced accuracy is also reported as the average class recall.

The DummyClassifier provides a simple reference point. A useful model should perform substantially better than this baseline.
""")

code("""fig, ax = plt.subplots(figsize=(11, 6))
plot_data = cv_results.melt(
    id_vars="Model",
    value_vars=["accuracy mean", "macro_f1 mean", "balanced_accuracy mean"],
    var_name="Metric", value_name="Mean score"
)
sns.barplot(data=plot_data, x="Model", y="Mean score", hue="Metric", ax=ax)
ax.set_ylim(0, 1)
ax.set_title("Repeated Cross-Validation Model Comparison")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(CHARTS / "07_cross_validation_comparison.png", dpi=180, bbox_inches="tight")
plt.show()
""")

md("## 12. Revised stratified holdout and hyperparameter tuning")

code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2026, stratify=y
)

print("Training records:", len(X_train))
print("Holdout records:", len(X_test))
print("\\nTraining class counts:")
display(y_train.value_counts().sort_index().to_frame("Training records"))
print("\\nHoldout class counts:")
display(y_test.value_counts().sort_index().to_frame("Holdout records"))
""")

md("""### Holdout interpretation

A new stratified 80:20 split is created for the revised tuning and final evaluation workflow. Hyperparameter selection uses only the training portion of this split. However, because the full dataset was used in previous project iterations, this holdout is not historically untouched. The repeated cross-validation results and this historical limitation must both be considered when interpreting performance.
""")

code("""tuning_pipeline = Pipeline([
    ("model", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1))
])

parameter_distributions = {
    "model__n_estimators": [150, 200, 250, 300],
    "model__max_depth": [None, 25, 35, 45],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.5]
}

search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=parameter_distributions,
    n_iter=8,
    scoring="f1_macro",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
    verbose=1
)

search.fit(X_train, y_train)

print("Best parameters:")
display(search.best_params_)
print("Best training CV macro F1:", round(search.best_score_, 5))
""")

code("""tuning_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
tuning_results.to_csv(REPORTS / "extra_trees_tuning_results.csv", index=False)
display(tuning_results[[
    "rank_test_score", "mean_test_score", "std_test_score",
    "mean_train_score", "params"
]].head(8))
""")

md("## 13. Final holdout evaluation")

code("""best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

final_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Macro F1": f1_score(y_test, y_pred, average="macro"),
    "Weighted F1": f1_score(y_test, y_pred, average="weighted"),
    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred)
}

final_metrics_df = pd.DataFrame(
    [{"Metric": k, "Score": v} for k, v in final_metrics.items()]
)
display(final_metrics_df.round(5))
final_metrics_df.to_csv(REPORTS / "final_model_metrics.csv", index=False)

print("Classification report:")
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).T
display(report_df.round(4))
report_df.to_csv(REPORTS / "classification_report.csv")
""")

md("## 14. Confusion matrix and class-level error analysis")

code("""labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
ax.set_title("Extra Trees - Holdout Confusion Matrix")
plt.tight_layout()
plt.savefig(CHARTS / "08_confusion_matrix.png", dpi=180, bbox_inches="tight")
plt.show()

class_errors = []
for i, label in enumerate(labels):
    row_total = cm[i].sum()
    correct = cm[i, i]
    class_errors.append({
        "Cover Type": label,
        "Correct predictions": int(correct),
        "Incorrect predictions": int(row_total - correct),
        "Recall": correct / row_total if row_total else np.nan
    })

class_error_df = pd.DataFrame(class_errors)
display(class_error_df.round(4))
class_error_df.to_csv(REPORTS / "class_error_analysis.csv", index=False)

off_diagonal = cm.copy()
np.fill_diagonal(off_diagonal, 0)
confusion_pairs = []
for i, actual in enumerate(labels):
    for j, predicted in enumerate(labels):
        if off_diagonal[i, j] > 0:
            confusion_pairs.append({
                "Actual class": actual,
                "Predicted class": predicted,
                "Misclassified records": int(off_diagonal[i, j])
            })
confusion_pairs_df = pd.DataFrame(confusion_pairs).sort_values(
    "Misclassified records", ascending=False
)
display(confusion_pairs_df.head(15))
confusion_pairs_df.to_csv(REPORTS / "most_common_class_confusions.csv", index=False)
""")

md("""### Error analysis interpretation

The confusion matrix shows correct and incorrect predictions for each actual and predicted class. Class-level recall identifies classes for which the model misses a larger proportion of observations. The most frequent off-diagonal pairs identify which cover types are most often confused. These results should guide any future feature engineering or data collection.
""")

md("## 15. Model interpretation: impurity and permutation importance")

code("""fitted_forest = best_model.named_steps["model"]
impurity_importance = pd.DataFrame({
    "Feature": X.columns,
    "Impurity importance": fitted_forest.feature_importances_
}).sort_values("Impurity importance", ascending=False)

display(impurity_importance.head(20).round(5))
impurity_importance.to_csv(REPORTS / "feature_importance.csv", index=False)

top_impurity = impurity_importance.head(20).sort_values("Impurity importance")
fig, ax = plt.subplots(figsize=(11, 8))
sns.barplot(data=top_impurity, x="Impurity importance", y="Feature", ax=ax)
ax.set_title("Top 20 Extra Trees Impurity-Based Feature Importances")
plt.tight_layout()
plt.savefig(CHARTS / "09_impurity_feature_importance.png", dpi=180, bbox_inches="tight")
plt.show()
""")

code("""permutation = permutation_importance(
    best_model, X_test, y_test,
    scoring="f1_macro", n_repeats=3,
    random_state=RANDOM_STATE, n_jobs=-1
)

permutation_df = pd.DataFrame({
    "Feature": X.columns,
    "Permutation importance mean": permutation.importances_mean,
    "Permutation importance std": permutation.importances_std
}).sort_values("Permutation importance mean", ascending=False)

display(permutation_df.head(20).round(5))
permutation_df.to_csv(REPORTS / "permutation_importance.csv", index=False)

top_perm = permutation_df.head(20).sort_values("Permutation importance mean")
fig, ax = plt.subplots(figsize=(11, 8))
sns.barplot(data=top_perm, x="Permutation importance mean", y="Feature", ax=ax)
ax.set_title("Top 20 Permutation Importances (Macro F1)")
plt.tight_layout()
plt.savefig(CHARTS / "10_permutation_importance.png", dpi=180, bbox_inches="tight")
plt.show()
""")

md("""### Feature importance interpretation

Impurity-based importance measures how much a feature contributes to splitting decisions within the fitted trees. It can favour variables with particular characteristics and can distribute importance across correlated predictors.

Permutation importance measures the reduction in a selected validation score when a feature's values are shuffled. Here the scoring metric is macro F1. Both measures describe model behaviour; neither establishes that a feature causes a forest cover type.
""")

md("## 16. Save the model and project outputs")

code("""model_path = MODELS / "extra_trees_forest_cover.joblib"
joblib.dump(best_model, model_path)

feature_schema = {
    "feature_names": list(X.columns),
    "target_name": target_col,
    "excluded_columns": excluded_columns,
    "model_type": "ExtraTreesClassifier inside scikit-learn Pipeline"
}
joblib.dump(feature_schema, MODELS / "feature_schema.joblib")

print("Saved model:", model_path)
print("Model size (MB):", round(model_path.stat().st_size / (1024 * 1024), 2))
print("Saved feature schema:", MODELS / "feature_schema.joblib")
print("\\nGenerated report files:")
for path in sorted(REPORTS.glob("*")):
    if path.is_file():
        print(path.name, "-", path.stat().st_size, "bytes")
print("\\nGenerated charts:")
for path in sorted(CHARTS.glob("*.png")):
    print(path.name, "-", path.stat().st_size, "bytes")
""")

md("## 17. Conclusions, limitations and future work")

code("""best_cv_row = cv_results.iloc[0]
lowest_recall_row = class_error_df.sort_values("Recall").iloc[0]
top_impurity_feature = impurity_importance.iloc[0]["Feature"]
top_perm_feature = permutation_df.iloc[0]["Feature"]

print("CONCLUSION")
print(f"- Best default candidate by mean CV macro F1: {best_cv_row['Model']}.")
print(f"- Mean CV macro F1: {best_cv_row['macro_f1 mean']:.4f} "
      f"(standard deviation: {best_cv_row['macro_f1 std']:.4f}).")
print(f"- Tuned Extra Trees holdout accuracy: {final_metrics['Accuracy']:.4f}.")
print(f"- Tuned Extra Trees holdout macro F1: {final_metrics['Macro F1']:.4f}.")
print(f"- Lowest-recall class in this holdout: {int(lowest_recall_row['Cover Type'])} "
      f"(recall: {lowest_recall_row['Recall']:.4f}).")
print(f"- Highest impurity importance: {top_impurity_feature}.")
print(f"- Highest permutation importance: {top_perm_feature}.")

print("\\nLIMITATIONS")
print("- The dataset was used in earlier project iterations; the revised holdout is not historically untouched.")
print("- Cross-validation scores are internal estimates on this dataset, not external geographic validation.")
print("- The dataset contains 15,120 records and may not represent every forest region or condition.")
print("- Feature importance describes model behaviour and does not establish causation.")
print("- Model performance may change when applied to data collected under different conditions.")

print("\\nFUTURE WORK")
print("- Evaluate on a genuinely new dataset collected independently of this project.")
print("- Investigate the most frequent class-confusion pairs.")
print("- Compare additional algorithms and carefully justified feature engineering.")
print("- Assess performance across geographic regions and different data collection conditions.")
""")

md("""## 18. Reproducibility and final checks

The notebook saves the fitted pipeline, feature schema, cross-validation results, tuning results, final metrics, classification report, class error analysis, feature importance tables and charts under the project folders.

For deployment or future inference, input data must contain the same predictor columns in the same logical schema used during training. The identifier and target must not be supplied as model predictors.
""")

nb = nbformat.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python"}
}
nbformat.validate(nb)
nbformat.write(nb, notebook_path)

print("NOTEBOOK CREATED:", notebook_path)
print("TOTAL CELLS:", len(cells))
print("MARKDOWN CELLS:", sum(c.cell_type == "markdown" for c in cells))
print("CODE CELLS:", sum(c.cell_type == "code" for c in cells))
