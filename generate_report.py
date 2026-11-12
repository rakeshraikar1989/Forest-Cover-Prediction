
from pathlib import Path
from datetime import datetime

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, PageBreak
)

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CHARTS = REPORTS / "charts"

output_file = REPORTS / "Forest_Cover_Prediction_Final_Report.pdf"

# Read actual project results
metrics = pd.read_csv(REPORTS / "model_metrics.csv")
comparison = pd.read_csv(REPORTS / "model_comparison.csv")
classification = pd.read_csv(
    REPORTS / "classification_report.csv", index_col=0
)
features = pd.read_csv(REPORTS / "feature_importance.csv")

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="CoverTitle",
    parent=styles["Title"],
    fontSize=24,
    leading=30,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#17365D"),
    spaceAfter=18
))

styles.add(ParagraphStyle(
    name="SectionHeading",
    parent=styles["Heading2"],
    fontSize=14,
    textColor=colors.HexColor("#17365D"),
    spaceBefore=12,
    spaceAfter=8
))

styles.add(ParagraphStyle(
    name="BodyTextCustom",
    parent=styles["BodyText"],
    fontSize=9,
    leading=14,
    spaceAfter=8
))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        0.65 * inch, 0.4 * inch,
        "Forest Cover Prediction | PRCP-1005"
    )
    canvas.drawRightString(
        7.85 * inch, 0.4 * inch,
        f"Page {doc.page}"
    )
    canvas.restoreState()

def make_table(data, widths=None):
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EAF0F6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table

def add_paragraph(story, text):
    story.append(Paragraph(text, styles["BodyTextCustom"]))

def add_chart(story, filename, width=6.2 * inch, max_height=5.5 * inch):
    chart_path = CHARTS / filename

    if chart_path.exists():
        chart = Image(str(chart_path))
        scale = min(
            width / chart.imageWidth,
            max_height / chart.imageHeight
        )

        chart.drawWidth = chart.imageWidth * scale
        chart.drawHeight = chart.imageHeight * scale

        story.append(chart)
        story.append(Spacer(1, 8))
    else:
        add_paragraph(story, f"Chart not found: {filename}")

story = []

# Cover page
story.append(Spacer(1, 1.2 * inch))
story.append(Paragraph(
    "FOREST COVER<br/>TYPE PREDICTION",
    styles["CoverTitle"]
))
story.append(Paragraph(
    "Machine Learning Classification Project",
    styles["Heading2"]
))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph(
    "<b>Project Code:</b> PRCP-1005",
    styles["BodyTextCustom"]
))
story.append(Paragraph(
    "<b>Prepared for:</b> DataMites Capstone Project",
    styles["BodyTextCustom"]
))
story.append(Paragraph(
    f"<b>Report date:</b> {datetime.now().strftime('%d %B %Y')}",
    styles["BodyTextCustom"]
))
story.append(Spacer(1, 0.4 * inch))
add_paragraph(
    story,
    "This project develops a machine learning classification model "
    "to predict forest cover types using environmental and "
    "cartographic features."
)
story.append(PageBreak())

# 1. Overview
story.append(Paragraph("1. Project Overview", styles["SectionHeading"]))
add_paragraph(
    story,
    "The objective is to predict one of seven forest cover types "
    "from environmental measurements, wilderness-area indicators "
    "and soil-type indicators."
)
add_paragraph(
    story,
    "The project uses a dataset containing 15,120 records and "
    "56 columns, including the identifier and target column. "
    "After removing the identifier and two constant features, "
    "52 predictor features were used for modelling."
)

# 2. Data preparation
story.append(Paragraph("2. Data Preparation", styles["SectionHeading"]))
add_paragraph(
    story,
    "The dataset was checked for missing values, duplicate records, "
    "constant columns and one-hot encoded wilderness and soil features. "
    "The Id column and two constant soil columns were excluded from "
    "model training. The target variable was Cover_Type."
)
add_paragraph(
    story,
    "The data was divided into training and testing subsets using "
    "an 80:20 stratified split with random_state=42."
)

# 3. Exploratory Data Analysis

story.append(Paragraph(
    "3. Exploratory Data Analysis",
    styles["SectionHeading"]
))

add_paragraph(
    story,
    "Exploratory Data Analysis (EDA) was performed to understand "
    "the distribution of forest cover types and the relationships "
    "between environmental features. The analysis included target "
    "class distribution, feature correlation, elevation distribution "
    "and elevation differences across forest cover types."
)

add_chart(story, "cover_type_distribution.png")

add_paragraph(
    story,
    "The dataset contains seven forest cover classes, with 2,160 "
    "records in each class. This balanced distribution provides "
    "equal representation of all target classes."
)

add_chart(story, "feature_correlation_heatmap.png")

add_paragraph(
    story,
    "The correlation heatmap was used to examine relationships "
    "between the ten continuous environmental features. Correlation "
    "indicates the strength and direction of linear relationships; "
    "it does not establish causation."
)

add_chart(story, "elevation_distribution.png")

add_paragraph(
    story,
    "Elevation distribution was examined to understand the range "
    "and frequency of elevation values across the dataset."
)

add_chart(story, "elevation_by_cover_type.png")

add_paragraph(
    story,
    "The elevation box plot compares elevation values across the "
    "seven forest cover types. It helps identify differences in "
    "elevation ranges and the spread of observations for each class."
)

# 4. Model Comparison
story.append(Paragraph("4. Model Comparison", styles["SectionHeading"]))

comparison_data = [["Model", "Accuracy", "Macro F1", "Training time"]]

for _, row in comparison.iterrows():
    comparison_data.append([
        str(row["Model"]),
        f'{row["Accuracy"]:.2%}',
        f'{row["Macro F1"]:.2%}',
        f'{row["Training Time (seconds)"]:.2f} sec'
    ])

story.append(make_table(
    comparison_data,
    [2.0 * inch, 1.1 * inch, 1.1 * inch, 1.5 * inch]
))

add_paragraph(
    story,
    "Extra Trees achieved the highest initial comparison accuracy "
    "of 85.65%, followed closely by Random Forest and "
    "Hist Gradient Boosting."
)

add_chart(story, "model_accuracy_comparison.png")

add_paragraph(
    story,
    "The model accuracy comparison chart shows the performance "
    "of the four machine learning algorithms. Extra Trees achieved "
    "the highest initial accuracy of 85.65%, followed closely by "
    "Random Forest at 85.42% and Hist Gradient Boosting at 85.22%. "
    "Decision Tree achieved 79.17% accuracy."
)

# 5. Hyperparameter Tuning and Final Evaluation
story.append(Paragraph(
    "5. Hyperparameter Tuning and Final Evaluation",
    styles["SectionHeading"]
))

metric_values = {
    row["Metric"]: float(row["Score"])
    for _, row in metrics.iterrows()
}

evaluation_data = [
    ["Evaluation Metric", "Result"],
    ["Final Test Accuracy", f'{metric_values["Test Accuracy"]:.2%}'],
    ["Cross-Validation Accuracy",
     f'{metric_values["Cross-Validation Accuracy"]:.2%}']
]

story.append(make_table(
    evaluation_data,
    [3.2 * inch, 2.0 * inch]
))

add_paragraph(
    story,
    "Extra Trees was tuned using RandomizedSearchCV with "
    "three-fold stratified cross-validation. The best cross-validation "
    "accuracy was approximately 83.78%. The final test accuracy "
    "was approximately 84.85%."
)

# 5. Classification report
story.append(Paragraph(
    "6. Classification Performance",
    styles["SectionHeading"]
))

class_names = {
    "1": "Spruce/Fir",
    "2": "Lodgepole Pine",
    "3": "Ponderosa Pine",
    "4": "Cottonwood/Willow",
    "5": "Aspen",
    "6": "Douglas-fir",
    "7": "Krummholz"
}

class_data = [[
    "Cover Type", "Precision", "Recall", "F1-score", "Support"
]]

for label, name in class_names.items():
    row = classification.loc[str(label)]
    class_data.append([
        f"{label} - {name}",
        f'{row["precision"]:.2%}',
        f'{row["recall"]:.2%}',
        f'{row["f1-score"]:.2%}',
        str(int(row["support"]))
    ])

story.append(make_table(
    class_data,
    [2.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 0.8 * inch]
))

add_paragraph(
    story,
    "The overall macro F1-score and weighted F1-score were both "
    f'{classification.loc["macro avg", "f1-score"]:.2%} and '
    f'{classification.loc["weighted avg", "f1-score"]:.2%}, respectively. '
    "Lodgepole Pine had the lowest recall among the seven classes."
)


# Classification Metrics Comparison

add_chart(story, "classification_metrics_comparison.png")

add_paragraph(
    story,
    "The classification metrics chart compares precision, recall "
    "and F1-score for each forest cover type. Classes 4 and 7 "
    "achieved particularly strong F1-scores, while Class 2 "
    "(Lodgepole Pine) had the lowest F1-score. This indicates "
    "that identifying Lodgepole Pine remains an area for improvement."
)

# 7. Confusion matrix
story.append(Paragraph("7. Confusion Matrix", styles["SectionHeading"]))

confusion_image = CHARTS / "confusion_matrix.png"
if confusion_image.exists():
    story.append(Image(str(confusion_image), width=6.2 * inch, height=4.2 * inch))
    story.append(Spacer(1, 8))

# 8. Feature Importance
story.append(Paragraph("8. Feature Importance", styles["SectionHeading"]))

top_features = features.head(15)
feature_data = [["Feature", "Importance"]]

for _, row in top_features.iterrows():
    feature_data.append([
        str(row["Feature"]),
        f'{row["Importance"]:.4f}'
    ])

story.append(make_table(
    feature_data,
    [3.5 * inch, 1.5 * inch]
))

add_paragraph(
    story,
    "Elevation was the most influential feature in the trained "
    "Extra Trees model, with an importance score of approximately "
    "0.1684. Other influential variables included wilderness area "
    "and horizontal distances to roads, fire points and hydrology."
)

importance_image = CHARTS / "feature_importance.png"
if importance_image.exists():
    story.append(Image(
        str(importance_image),
        width=6.2 * inch,
        height=4.0 * inch
    ))

# 9. Limitations
story.append(Paragraph("9. Limitations", styles["SectionHeading"]))
add_paragraph(
    story,
    "The initial model comparison used the test split to compare "
    "candidate models. The tuned Extra Trees model was subsequently "
    "evaluated on that same test split. Therefore, the final test "
    "accuracy is not a fully independent estimate of generalisation "
    "performance and may be affected by model-selection bias. "
    "A separate untouched holdout dataset is recommended for a "
    "more rigorous final evaluation."
)

# 9. Conclusion
story.append(Paragraph("10. Conclusion and Future Improvements",
                       styles["SectionHeading"]))
add_paragraph(
    story,
    "The project successfully developed an Extra Trees classifier "
    "for seven forest cover classes, achieving 84.85% test accuracy. "
    "Future work could focus on improving recall for Lodgepole Pine, "
    "using an independent holdout dataset, testing additional models "
    "and evaluating the model on a larger dataset."
)

doc = SimpleDocTemplate(
    str(output_file),
    pagesize=A4,
    rightMargin=0.6 * inch,
    leftMargin=0.6 * inch,
    topMargin=0.65 * inch,
    bottomMargin=0.7 * inch
)

doc.build(story, onFirstPage=footer, onLaterPages=footer)

print("PDF report created successfully!")
print("Location:", output_file)
print("File size:", f"{output_file.stat().st_size:,} bytes")
