"""Export cleaned sensor and model outputs to a Power BI-ready workbook.

Run from the project root:
    python export_powerbi.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
DEST = ROOT / "PredictiveMaintenance_PowerBI_v2.xlsx"

raw = pd.read_csv(DATA / "cleaned_sensor_data.csv", parse_dates=["Timestamp"])
preds = pd.read_csv(OUTPUTS / "test_predictions.csv", parse_dates=["Timestamp"])
fi = pd.read_csv(OUTPUTS / "feature_importance.csv")
metrics = pd.read_csv(OUTPUTS / "model_comparison.csv")

latest = raw.sort_values("Timestamp").groupby("Machine_ID").tail(1).copy()
latest["Observed_Failure_Rate"] = raw.groupby("Machine_ID")["Failure"].mean().reindex(latest["Machine_ID"]).to_numpy()
latest_pred = preds.sort_values("Timestamp").groupby("Machine_ID").tail(1)[["Machine_ID", "Failure_Probability", "Predicted_Failure"]]

# Left-merge: machines used for training (not in test_predictions.csv) will have
# NaN for Failure_Probability and Predicted_Failure. These are labelled
# 'Not Scored' in Risk_Status and also written to a dedicated 'UnscoredMachines'
# sheet so Power BI users can clearly distinguish scored vs. training-set machines.
summary = latest.merge(latest_pred, on="Machine_ID", how="left")
summary["Risk_Status"] = np.select(
    [summary["Failure_Probability"].isna(), summary["Failure_Probability"].fillna(0) > 0.5, summary["Failure_Probability"].fillna(0) > 0.2],
    ["Not Scored", "High Risk", "Warning"], default="Healthy"
)
summary = summary.rename(columns={"Failure": "Latest_Observed_Failure"})

# Separate scored and unscored machines for transparency
scored_summary   = summary[summary["Risk_Status"] != "Not Scored"].copy()
unscored_summary = summary[summary["Risk_Status"] == "Not Scored"].copy()
daily = raw.set_index("Timestamp").resample("D").agg(
    Failure_Events=("Failure", "sum"), Avg_Temperature=("Temperature", "mean"),
    Avg_Vibration=("Vibration", "mean"), Avg_Pressure=("Pressure", "mean"), Avg_RPM=("RPM", "mean")
).reset_index()
top = preds.sort_values("Failure_Probability", ascending=False).head(500)

with pd.ExcelWriter(DEST, engine="openpyxl") as writer:
    summary.to_excel(writer, index=False, sheet_name="MachineSummary")
    scored_summary.to_excel(writer, index=False, sheet_name="ScoredMachines")
    unscored_summary.to_excel(writer, index=False, sheet_name="UnscoredMachines")
    daily.to_excel(writer, index=False, sheet_name="SensorDailyTrend")
    top.to_excel(writer, index=False, sheet_name="TopHighRiskReadings")
    fi.to_excel(writer, index=False, sheet_name="FeatureImportance")
    metrics.to_excel(writer, index=False, sheet_name="ModelComparison")

wb = load_workbook(DEST)
for ws in wb.worksheets:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="17365D")
        cell.alignment = Alignment(wrap_text=True)
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        maxlen = max((len(str(c.value)) if c.value is not None else 0) for c in list(col)[:200])
        ws.column_dimensions[letter].width = min(max(maxlen + 2, 12), 32)
wb.save(DEST)
print(f"Created {DEST}")
