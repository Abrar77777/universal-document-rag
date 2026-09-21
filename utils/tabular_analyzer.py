from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _safe_float(value: Any):
    if pd.isna(value):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _read_tables(file_path: str) -> dict[str, pd.DataFrame]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return {"data": pd.read_csv(file_path)}
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=None)
    return {}


def analyze_tabular_file(file_path: str) -> dict[str, Any]:
    tables = _read_tables(file_path)
    if not tables:
        return {}

    profile: dict[str, Any] = {
        "file": Path(file_path).name,
        "sheets": [],
        "chart_specs": [],
        "summary_text": "",
    }
    summary_lines = []

    for sheet_name, df in tables.items():
        cleaned = df.dropna(how="all")
        numeric_cols = cleaned.select_dtypes(include="number").columns.tolist()
        categorical_cols = cleaned.select_dtypes(exclude="number").columns.tolist()
        date_cols = [
            col for col in cleaned.columns
            if "date" in str(col).lower() or pd.api.types.is_datetime64_any_dtype(cleaned[col])
        ]

        sheet_profile = {
            "name": sheet_name,
            "rows": int(cleaned.shape[0]),
            "columns": int(cleaned.shape[1]),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_values": {str(k): int(v) for k, v in cleaned.isna().sum().items()},
            "missing_percent": {
                str(k): _safe_float((v / max(len(cleaned), 1)) * 100)
                for k, v in cleaned.isna().sum().items()
            },
            "numeric_summary": {},
            "top_categories": {},
            "correlations": [],
            "outliers": [],
        }

        for col in numeric_cols:
            series = cleaned[col].dropna()
            sheet_profile["numeric_summary"][col] = {
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "mean": _safe_float(series.mean()),
                "median": _safe_float(series.median()),
                "std": _safe_float(series.std()),
            }
            if not series.empty:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr:
                    outlier_count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
                    if outlier_count:
                        sheet_profile["outliers"].append({
                            "column": col,
                            "count": outlier_count,
                            "lower_bound": _safe_float(q1 - 1.5 * iqr),
                            "upper_bound": _safe_float(q3 + 1.5 * iqr),
                        })

        for col in categorical_cols[:8]:
            sheet_profile["top_categories"][col] = (
                cleaned[col].astype(str).value_counts().head(10).to_dict()
            )

        if len(numeric_cols) >= 2:
            corr = cleaned[numeric_cols].corr(numeric_only=True).round(4)
            for left in corr.columns:
                for right in corr.columns:
                    if left != right:
                        sheet_profile["correlations"].append({
                            "x": left,
                            "y": right,
                            "correlation": _safe_float(corr.loc[left, right]),
                        })
            strongest = sorted(
                sheet_profile["correlations"],
                key=lambda item: abs(item["correlation"] or 0),
                reverse=True,
            )[:20]
            sheet_profile["correlations"] = strongest

        if numeric_cols:
            first_num = numeric_cols[0]
            hist = cleaned[first_num].dropna()
            hist_df = (
                pd.cut(hist, bins=min(10, max(1, hist.nunique())))
                .value_counts()
                .sort_index()
                .reset_index(name="count")
            )
            hist_df[first_num] = hist_df[first_num].astype(str)
            profile["chart_specs"].append({
                "title": f"{sheet_name}: distribution of {first_num}",
                "type": "histogram",
                "sheet": sheet_name,
                "x": first_num,
                "data": hist_df.to_dict(orient="records"),
            })
            if categorical_cols:
                first_cat = categorical_cols[0]
                grouped = (
                    cleaned.groupby(first_cat, dropna=True)[first_num]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                    .reset_index()
                )
                profile["chart_specs"].append({
                    "title": f"{sheet_name}: average {first_num} by {first_cat}",
                    "type": "bar",
                    "sheet": sheet_name,
                    "x": first_cat,
                    "y": first_num,
                    "aggregation": "mean",
                    "data": grouped.to_dict(orient="records"),
                })
            if len(numeric_cols) >= 2:
                second_num = numeric_cols[1]
                scatter = cleaned[[first_num, second_num]].dropna().head(1000)
                profile["chart_specs"].append({
                    "title": f"{sheet_name}: {first_num} vs {second_num}",
                    "type": "scatter",
                    "sheet": sheet_name,
                    "x": first_num,
                    "y": second_num,
                    "data": scatter.to_dict(orient="records"),
                })
            if date_cols:
                date_col = date_cols[0]
                time_df = cleaned.copy()
                time_df[date_col] = pd.to_datetime(time_df[date_col], errors="coerce")
                grouped = (
                    time_df.dropna(subset=[date_col])
                    .groupby(date_col)[first_num]
                    .sum()
                    .reset_index()
                    .sort_values(date_col)
                    .tail(30)
                )
                grouped[date_col] = grouped[date_col].astype(str)
                profile["chart_specs"].append({
                    "title": f"{sheet_name}: {first_num} over {date_col}",
                    "type": "line",
                    "sheet": sheet_name,
                    "x": date_col,
                    "y": first_num,
                    "aggregation": "sum",
                    "data": grouped.to_dict(orient="records"),
                })

        summary_lines.append(
            f"{sheet_name}: {sheet_profile['rows']} rows, {sheet_profile['columns']} columns, "
            f"numeric columns: {', '.join(numeric_cols) or 'none'}."
        )
        profile["sheets"].append(sheet_profile)

    profile["summary_text"] = "\n".join(summary_lines)
    return profile


def load_sheet_for_chart(file_path: str, sheet_name: str) -> pd.DataFrame:
    tables = _read_tables(file_path)
    return tables.get(sheet_name, pd.DataFrame())
