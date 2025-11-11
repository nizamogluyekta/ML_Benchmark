"""Utility helpers for persisting metrics, plots, and textual summaries."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import json
import math

import pandas as pd

try:
  import matplotlib
  matplotlib.use("Agg", force=True)
  import matplotlib.pyplot as plt
  import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
  plt = None
  sns = None
  PLOTTING_AVAILABLE = False
else:  # pragma: no cover - import side effects
  PLOTTING_AVAILABLE = True
  plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12})


DEFAULT_ERROR_THRESHOLD = 0.05


@dataclass
class MetricResult:
  """Structured representation of a single model evaluation."""
  dataset: str
  model: str
  metric: str
  value: float


def ensure_dir(path: Path) -> Path:
  path.mkdir(parents=True, exist_ok=True)
  return path


def _fmt_metric(value: Optional[float]) -> str:
  if value is None or (isinstance(value, float) and math.isnan(value)):
    return "-"
  return f"{value:.4f}"


def save_metrics_csv(metrics: Iterable[MetricResult], output_file: Path) -> None:
  """Write metrics to disk in a normalized long format."""
  ensure_dir(output_file.parent)
  rows = [asdict(metric) for metric in metrics]
  if not rows:
    return
  with output_file.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def save_metrics_json(metrics: Iterable[MetricResult], output_file: Path) -> None:
  """Persist metrics to JSON for downstream aggregation."""
  ensure_dir(output_file.parent)
  rows = [asdict(metric) for metric in metrics]
  with output_file.open("w") as handle:
    json.dump(rows, handle, indent=2)


def write_run_summary(summary: Dict[str, str], output_file: Path) -> None:
  """Persist a human-readable summary for the run."""
  ensure_dir(output_file.parent)
  with output_file.open("w") as handle:
    for key, value in summary.items():
      handle.write(f"{key}: {value}\n")


def report_placeholder(output_dir: Path, dataset: str, model: str) -> None:
  """Drop a minimal marker file so pipelines can verify execution order."""
  ensure_dir(output_dir)
  marker = output_dir / "_SUCCESS"
  marker.write_text(f"dataset={dataset}\nmodel={model}\n")


def predictions_to_frame(
  test_df: pd.DataFrame,
  predictions: Sequence[float],
  dataset_cfg: Dict[str, str],
) -> pd.DataFrame:
  """Return a dataframe with date, actual, and predicted values."""
  target = dataset_cfg.get("target")
  if not target:
    raise KeyError("Dataset configuration must specify a target column")
  datetime_col = dataset_cfg.get("datetime", "DATE_KEY")
  pred_series = pd.Series(predictions, index=test_df.index[: len(predictions)])
  frame = pd.DataFrame({
    "DATE_KEY": test_df[datetime_col].iloc[: len(pred_series)].values if datetime_col in test_df else range(len(pred_series)),
    "Actual": test_df[target].iloc[: len(pred_series)].values,
    "Predicted": pred_series.values,
  })
  frame["DATE_KEY"] = pd.to_datetime(frame["DATE_KEY"])
  return frame


def enrich_predictions_frame(df: pd.DataFrame, error_threshold: float = DEFAULT_ERROR_THRESHOLD) -> pd.DataFrame:
  """Add error-related columns required for visualization/reporting."""
  enriched = df.copy()
  enriched["Error"] = enriched["Predicted"] - enriched["Actual"]
  denominator = enriched["Actual"].abs().replace(0, pd.NA)
  enriched["Percent_Error"] = ((enriched["Error"].abs() / denominator) * 100).astype(float)
  enriched["Within_Threshold"] = (enriched["Percent_Error"] <= (error_threshold * 100)).fillna(False)
  return enriched


def metrics_to_dict(metrics: Iterable[MetricResult]) -> Dict[str, float]:
  return {metric.metric: metric.value for metric in metrics}


def _plot_actual_vs_predicted(df: pd.DataFrame, output_path: Path, title: str) -> None:
  plt.style.use("seaborn-v0_8-whitegrid")
  plt.figure(figsize=(12, 6))
  plt.plot(df["DATE_KEY"], df["Actual"], "b-", label="Actual", linewidth=2)
  plt.plot(df["DATE_KEY"], df["Predicted"], "r--", label="Predicted", linewidth=2)
  plt.title(title)
  plt.xlabel("Date")
  plt.ylabel("Target Value")
  plt.legend()
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.savefig(output_path, bbox_inches="tight")
  plt.close()


def _plot_scatter(df: pd.DataFrame, output_path: Path, r2: float) -> None:
  plt.style.use("seaborn-v0_8-whitegrid")
  plt.figure(figsize=(10, 8))
  plt.scatter(df["Actual"], df["Predicted"], alpha=0.6)
  max_val = max(df["Actual"].max(), df["Predicted"].max())
  min_val = min(df["Actual"].min(), df["Predicted"].min())
  plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2)
  plt.title("Actual vs Predicted")
  plt.xlabel("Actual")
  plt.ylabel("Predicted")
  plt.annotate(
    f"R² = {r2:.4f}",
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
  )
  plt.tight_layout()
  plt.savefig(output_path, bbox_inches="tight")
  plt.close()


def _plot_error_hist(df: pd.DataFrame, output_path: Path) -> None:
  plt.figure(figsize=(10, 6))
  sns.histplot(df["Error"], kde=True)
  plt.axvline(x=0, color="r", linestyle="--")
  plt.title("Distribution of Prediction Errors")
  plt.xlabel("Prediction Error")
  plt.ylabel("Frequency")
  plt.tight_layout()
  plt.savefig(output_path, bbox_inches="tight")
  plt.close()


def _plot_error_over_time(df: pd.DataFrame, output_path: Path, threshold_pct: float) -> None:
  plt.figure(figsize=(12, 6))
  plt.bar(df["DATE_KEY"], df["Percent_Error"], color="skyblue")
  plt.axhline(y=threshold_pct, color="r", linestyle="--", label=f"{threshold_pct:.1f}% Threshold")
  plt.title("Percentage Error by Date")
  plt.xlabel("Date")
  plt.ylabel("Percent Error (%)")
  plt.legend()
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.savefig(output_path, bbox_inches="tight")
  plt.close()


def _plot_within_threshold(df: pd.DataFrame, output_path: Path, within_pct: float) -> None:
  plt.figure(figsize=(8, 8))
  labels = ["Within Threshold", "Outside Threshold"]
  sizes = [within_pct, max(0, 100 - within_pct)]
  colors = ["#66b3ff", "#ff9999"]
  explode = (0.1, 0)
  plt.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    shadow=True,
    startangle=90,
  )
  plt.axis("equal")
  plt.title("Predictions Within Error Threshold")
  plt.tight_layout()
  plt.savefig(output_path, bbox_inches="tight")
  plt.close()


def _plot_monthly_error(df: pd.DataFrame, output_path: Path) -> None:
  month_df = df.copy()
  month_df["Month"] = month_df["DATE_KEY"].dt.month
  monthly_error = month_df.groupby("Month")["Percent_Error"].mean().reset_index()
  plt.figure(figsize=(10, 6))
  sns.barplot(x="Month", y="Percent_Error", data=monthly_error, color="steelblue")
  plt.title("Average Percentage Error by Month")
  plt.xlabel("Month")
  plt.ylabel("Average Percent Error (%)")
  plt.tight_layout()
  plt.savefig(output_path, bbox_inches="tight")
  plt.close()


def create_performance_visuals(
  enriched_df: pd.DataFrame,
  metrics: Dict[str, float],
  output_dir: Path,
  model_label: str,
  file_prefix: str,
  threshold_pct: float,
) -> List[Path]:
  if not PLOTTING_AVAILABLE:
    return []
  ensure_dir(output_dir)
  generated_paths: List[Path] = []
  r2 = metrics.get("r2", math.nan)
  within_pct = enriched_df["Within_Threshold"].mean() * 100

  avp = output_dir / f"{file_prefix}_actual_vs_predicted.png"
  _plot_actual_vs_predicted(enriched_df, avp, f"{model_label}: Actual vs Predicted")
  generated_paths.append(avp)

  scatter = output_dir / f"{file_prefix}_scatter_plot.png"
  _plot_scatter(enriched_df, scatter, r2=r2 if not math.isnan(r2) else 0.0)
  generated_paths.append(scatter)

  err_hist = output_dir / f"{file_prefix}_error_distribution.png"
  _plot_error_hist(enriched_df, err_hist)
  generated_paths.append(err_hist)

  err_time = output_dir / f"{file_prefix}_error_over_time.png"
  _plot_error_over_time(enriched_df, err_time, threshold_pct)
  generated_paths.append(err_time)

  within_plot = output_dir / f"{file_prefix}_within_threshold.png"
  _plot_within_threshold(enriched_df, within_plot, within_pct)
  generated_paths.append(within_plot)

  monthly = output_dir / f"{file_prefix}_monthly_error.png"
  _plot_monthly_error(enriched_df, monthly)
  generated_paths.append(monthly)

  return generated_paths


def generate_performance_report(
  enriched_df: pd.DataFrame,
  metrics: Dict[str, float],
  output_dir: Path,
  model_label: str,
  file_prefix: str,
  threshold_pct: float,
  extra_sections: Optional[Iterable[Tuple[str, str]]] = None,
) -> Path:
  ensure_dir(output_dir)
  within_pct = enriched_df["Within_Threshold"].mean() * 100
  rmse = metrics.get("rmse")
  mae = metrics.get("mae")
  r2 = metrics.get("r2")
  mse = metrics.get("mse", rmse ** 2 if rmse is not None else None)
  mape = metrics.get("mape")
  smape = metrics.get("smape")
  median_ae = metrics.get("median_ae")
  p90_ae = metrics.get("p90_ae")
  bias = metrics.get("bias")
  coverage5 = metrics.get("coverage_within_5pct")
  coverage10 = metrics.get("coverage_within_10pct")
  rmsle = metrics.get("rmsle")
  train_time = metrics.get("train_time_sec")
  predict_time = metrics.get("predict_time_sec")

  monthly_error = (
    enriched_df.groupby(enriched_df["DATE_KEY"].dt.month)["Percent_Error"].agg(["mean", "min", "max"]).reset_index()
  )
  monthly_error.columns = ["Month", "Average Error (%)", "Min Error (%)", "Max Error (%)"]

  detailed = enriched_df.copy()
  detailed["DATE_KEY"] = detailed["DATE_KEY"].dt.strftime("%Y-%m-%d")
  detailed = detailed.rename(
    columns={
      "DATE_KEY": "Date",
      "Actual": "Actual Value",
      "Predicted": "Predicted Value",
      "Error": "Error",
      "Percent_Error": "Error (%)",
      "Within_Threshold": f"Within {threshold_pct:.1f}%",
    }
  )
  for col in ["Actual Value", "Predicted Value", "Error", "Error (%)"]:
    detailed[col] = detailed[col].round(3)
  detailed[f"Within {threshold_pct:.1f}%"] = detailed[f"Within {threshold_pct:.1f}%"].map({True: "Yes", False: "No"})

  report_lines: List[str] = []
  report_lines.append(f"# Model Performance Report – {model_label}")
  report_lines.append("")
  report_lines.append("## Summary Metrics")
  if mse is not None:
    report_lines.append(f"- Mean Squared Error (MSE): {mse:.4f}")
  if rmse is not None:
    report_lines.append(f"- Root Mean Squared Error (RMSE): {rmse:.4f}")
  if mae is not None:
    report_lines.append(f"- Mean Absolute Error (MAE): {mae:.4f}")
  if r2 is not None:
    report_lines.append(f"- R² Score: {r2:.4f}")
  if mape is not None:
    report_lines.append(f"- Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
  if smape is not None:
    report_lines.append(f"- Symmetric MAPE: {smape:.2f}%")
  if median_ae is not None:
    report_lines.append(f"- Median Absolute Error: {median_ae:.4f}")
  if p90_ae is not None:
    report_lines.append(f"- 90th Percentile Absolute Error: {p90_ae:.4f}")
  if rmsle is not None:
    report_lines.append(f"- Root Mean Squared Log Error (RMSLE): {rmsle:.4f}")
  if bias is not None:
    report_lines.append(f"- Mean Bias (Predicted - Actual): {bias:.4f}")
  if coverage5 is not None:
    report_lines.append(f"- Coverage within 5%: {coverage5:.2f}%")
  if coverage10 is not None:
    report_lines.append(f"- Coverage within 10%: {coverage10:.2f}%")
  report_lines.append(f"- Predictions within threshold ({threshold_pct:.1f}%): {within_pct:.2f}%")
  if train_time is not None:
    report_lines.append(f"- Training Time: {train_time:.3f}s")
  if predict_time is not None:
    report_lines.append(f"- Prediction Time: {predict_time:.3f}s")
  report_lines.append("")

  extended = [
    ("MAPE (%)", mape),
    ("sMAPE (%)", smape),
    ("Median AE", median_ae),
    ("P90 AE", p90_ae),
    ("Bias", bias),
    ("Coverage ≤5%", coverage5),
    ("Coverage ≤10%", coverage10),
    ("RMSLE", rmsle),
    ("Train Time (s)", train_time),
    ("Predict Time (s)", predict_time),
  ]
  if any(val is not None for _, val in extended):
    report_lines.append("### Extended Metrics")
    report_lines.append("| Metric | Value |")
    report_lines.append("| --- | --- |")
    for label, val in extended:
      report_lines.append(f"| {label} | {_fmt_metric(val)} |")
    report_lines.append("")

  report_lines.append("## Monthly Error Profile")
  report_lines.append(monthly_error.to_string(index=False))
  report_lines.append("")

  report_lines.append("## Detailed Predictions")
  report_lines.append(detailed.to_string(index=False))

  if extra_sections:
    for title, content in extra_sections:
      report_lines.append("")
      report_lines.append(f"## {title}")
      report_lines.append(content)

  report_path = output_dir / f"{file_prefix}_performance_report.md"
  report_path.write_text("\n".join(report_lines), encoding="utf-8")
  return report_path


def generate_detailed_report(
  predictions_df: pd.DataFrame,
  metrics: Iterable[MetricResult],
  output_dir: Path,
  model_label: str,
  file_prefix: str,
  error_threshold: float = DEFAULT_ERROR_THRESHOLD,
  extra_sections: Optional[Iterable[Tuple[str, str]]] = None,
) -> Dict[str, List[Path] | Path]:
  enriched = enrich_predictions_frame(predictions_df, error_threshold)
  metrics_map = metrics_to_dict(metrics)
  threshold_pct = error_threshold * 100
  sections: List[Tuple[str, str]] = list(extra_sections or [])

  if PLOTTING_AVAILABLE:
    figures = create_performance_visuals(
      enriched_df=enriched,
      metrics=metrics_map,
      output_dir=output_dir,
      model_label=model_label,
      file_prefix=file_prefix,
      threshold_pct=threshold_pct,
    )
  else:
    figures = []
    sections.append(
      (
        "Visualizations",
        "Plots could not be generated because matplotlib/seaborn are not installed."
        " Install them and re-run the benchmark to obtain figures.",
      )
    )
  report_path = generate_performance_report(
    enriched_df=enriched,
    metrics=metrics_map,
    output_dir=output_dir,
    model_label=model_label,
    file_prefix=file_prefix,
    threshold_pct=threshold_pct,
    extra_sections=sections,
  )
  return {"report": report_path, "figures": figures}
