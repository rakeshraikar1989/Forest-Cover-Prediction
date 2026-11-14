from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

R = Path("reports")
C = R / "charts"
m = pd.read_csv(R / "final_model_metrics.csv")
cv = pd.read_csv(R / "cross_validation_model_comparison.csv")
cl = pd.read_csv(R / "classification_report.csv", index_col=0)
er = pd.read_csv(R / "class_error_analysis.csv")
cf = pd.read_csv(R / "most_common_class_confusions.csv")
pi = pd.read_csv(R / "permutation_importance.csv")
v = dict(zip(m["Metric"], m["Score"]))

s = getSampleStyleSheet()
s.add(ParagraphStyle(name="Body", parent=s["BodyText"], fontSize=9, leading=13, spaceAfter=7))
story = []

def p(x, style="Body"):
    story.append(Paragraph(str(x), s[style]))

def h(x):
    p(x, "Heading1")

def t(rows, widths=None):
    q = Table(rows, colWidths=widths, repeatRows=1)
    q.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#243B53")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("GRID",(0,0),(-1,-1),0.4,colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF2F6")]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5)
    ]))
    story.extend([q, Spacer(1,10)])

def img(name):
    f = C / name
    if f.exists():
        q = Image(str(f))
        q.drawWidth = 16*cm
        q.drawHeight = q.drawWidth*q.imageHeight/q.imageWidth
        q.hAlign = "CENTER"
        story.extend([q, Spacer(1,8)])

p("FOREST COVER TYPE PREDICTION", "Title")
p("Machine Learning Classification Project | PRCP-1005", "Heading2")
p("Final model: Tuned Extra Trees")
p(f"Holdout accuracy: {v['Accuracy']:.2%} | Macro F1: {v['Macro F1']:.2%}")
story.append(PageBreak())

h("1. Executive Summary")
p("This project predicts seven forest cover types from cartographic and environmental measurements. The dataset contains 15,120 records and 56 original columns. After removing the identifier and two constant columns, 52 predictors were retained.")
t([["Evaluation metric","Result"],
   ["Accuracy",f"{v['Accuracy']:.2%}"],
   ["Macro F1",f"{v['Macro F1']:.2%}"],
   ["Weighted F1",f"{v['Weighted F1']:.2%}"],
   ["Balanced accuracy",f"{v['Balanced Accuracy']:.2%}"]])

h("2. Dataset and Data Quality")
p("The dataset has seven equally represented target classes, with 2,160 records per class. The audit found no missing values and no duplicate rows. Soil_Type7 and Soil_Type15 were constant zero columns and were removed. Id was excluded because it is an identifier.")
img("01_target_distribution.png")

h("3. Preprocessing Decisions")
p("No imputation was required because there were no missing values. Soil and wilderness variables were already represented by binary indicators, so additional one-hot encoding was unnecessary. Scaling was not applied because the candidate algorithms were tree-based. A scikit-learn Pipeline was used for model fitting.")
img("02_indicator_frequency.png")

h("4. Exploratory Data Analysis")
p("EDA covered numerical feature distributions, descriptive statistics, skewness, IQR-based potential outliers, correlations, class-level median profiles and elevation differences across cover types.")
for f in ["03_numeric_feature_distributions.png","04_continuous_feature_correlation.png","05_feature_target_median_heatmap.png","06_elevation_by_cover_type.png"]:
    img(f)
p("Potential outliers were flagged using the 1.5 × IQR rule and retained because environmental extremes may be valid observations. Correlation does not establish causation.")

h("5. Model Comparison and Validation")
p("Decision Tree, Random Forest, Extra Trees and Hist Gradient Boosting were compared using repeated stratified three-fold cross-validation with two repeats. The reported standard deviations describe variation across six folds, not confidence intervals.")
rows = [["Model","Accuracy mean","Macro F1 mean","Macro F1 SD"]]
for _,r in cv.iterrows():
    rows.append([r["Model"],f"{r['accuracy mean']:.2%}",f"{r['macro_f1 mean']:.2%}",f"{r['macro_f1 std']:.4f}"])
t(rows)
best = cv.sort_values("macro_f1 mean",ascending=False).iloc[0]
p(f"Extra Trees was the strongest default candidate by mean CV macro F1: {best['macro_f1 mean']:.2%} (SD {best['macro_f1 std']:.4f}). A DummyClassifier baseline was also evaluated in the notebook.")
img("07_cross_validation_comparison.png")

h("6. Hyperparameter Tuning")
p("Extra Trees was tuned using RandomizedSearchCV, three-fold stratified cross-validation and macro F1 as the selection metric. Tuning was performed on the training portion of a stratified 80:20 split. Search results are saved in reports/extra_trees_tuning_results.csv.")

h("7. Holdout Results")
p("The holdout contains 3,024 records, with 432 records per class.")
rows = [["Class","Precision","Recall","F1","Support"]]
for label in ["1","2","3","4","5","6","7"]:
    if label in cl.index:
        r=cl.loc[label]
        rows.append([label,f"{r['precision']:.2%}",f"{r['recall']:.2%}",f"{r['f1-score']:.2%}",int(r["support"])])
t(rows)
img("08_confusion_matrix.png")

h("8. Error Analysis")
weak = er.sort_values("Recall").iloc[0]
strong = er.sort_values("Recall",ascending=False).iloc[0]
p(f"Class {int(weak['Cover Type'])} had the lowest recall ({weak['Recall']:.2%}); class {int(strong['Cover Type'])} had the highest ({strong['Recall']:.2%}).")
rows=[["Actual class","Predicted class","Errors"]]
for _,r in cf.head(8).iterrows():
    rows.append([int(r["Actual class"]),int(r["Predicted class"]),int(r["Misclassified records"])])
t(rows)

h("9. Feature Interpretation")
p("Impurity-based importance and permutation importance were calculated. Permutation importance used macro F1 on the holdout. These measures describe model behaviour, not causal effects.")
img("09_impurity_feature_importance.png")
img("10_permutation_importance.png")
rows=[["Feature","Permutation importance","SD"]]
for _,r in pi.head(10).iterrows():
    rows.append([r["Feature"],f"{r['Permutation importance mean']:.5f}",f"{r['Permutation importance std']:.5f}"])
t(rows)

h("10. Limitations")
for x in [
"The dataset was used in earlier project iterations; the revised holdout is not historically untouched.",
"Cross-validation and holdout results are internal estimates, not external geographic validation.",
"The dataset may not represent all forest regions or data collection conditions.",
"Class 2 has the lowest recall and needs further investigation.",
"Feature importance does not establish causation."
]:
    p("• "+x)

h("11. Conclusion and Future Work")
p(f"The tuned Extra Trees model achieved {v['Accuracy']:.2%} holdout accuracy and {v['Macro F1']:.2%} macro F1. Repeated cross-validation also selected Extra Trees as the strongest default candidate. Future work should evaluate an independently collected dataset, investigate frequent class confusions and assess geographic generalisation.")

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica",8)
    canvas.drawString(1.7*cm,1.0*cm,"Forest Cover Prediction | PRCP-1005")
    canvas.drawRightString(A4[0]-1.7*cm,1.0*cm,f"Page {doc.page}")
    canvas.restoreState()

out=R/"Forest_Cover_Prediction_Final_Report.pdf"
SimpleDocTemplate(str(out),pagesize=A4,rightMargin=1.7*cm,leftMargin=1.7*cm,topMargin=1.6*cm,bottomMargin=1.6*cm).build(story,onFirstPage=footer,onLaterPages=footer)
print("PDF CREATED:",out)
print("FILE SIZE:",out.stat().st_size,"bytes")
