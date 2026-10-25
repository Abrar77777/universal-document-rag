"""Conservative whole-column extrema queries executed by pandas, never by eval."""
import math
import re

import pandas as pd


def words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


PRICE_WORDS = {"price", "prices", "cost", "costs", "costliest", "expensive", "cheapest", "priciest"}
PRICE_COLUMNS = {"price", "cost", "unit price", "unit cost", "selling price"}
HIGH_WORDS = {"highest", "maximum", "max", "largest", "greatest", "costliest", "priciest"}
LOW_WORDS = {"lowest", "minimum", "min", "smallest", "cheapest"}
FILLER = set("which what is are the a an has have having item items product products record records row rows show me find tell with of value values costs more most least expensive price prices cost costliest priciest cheapest".split())


def _clarification(message):
    return {"answer": message, "generation_status": "clarification", "table_evidence": []}


def answer_table_question(question, sources):
    tokens = set(words(question))
    high = bool(tokens & HIGH_WORDS or "expensive" in tokens or (tokens & PRICE_WORDS and tokens & {"more", "most"}))
    low = bool(tokens & LOW_WORDS or (tokens & PRICE_WORDS and "least" in tokens))
    if "least" in tokens and "expensive" in tokens:
        high = False
    if not sources or high == low:
        return None
    candidates = []
    for source in sources:
        frame = source["data"]
        for column in frame.select_dtypes(include="number").columns:
            column_words = words(column)
            exact = set(column_words) <= tokens
            price_alias = bool(tokens & PRICE_WORDS) and " ".join(column_words) in PRICE_COLUMNS
            if exact or price_alias:
                candidates.append((source, column, set(column_words)))
    if not candidates:
        return None
    allowed = FILLER | HIGH_WORDS | LOW_WORDS
    for _, _, column_words in candidates:
        allowed |= column_words
    # Do not silently drop filters, requested counts, comparisons, or named entities.
    if tokens - allowed:
        return None
    if len(candidates) > 1:
        choices = "; ".join(f"{source['source']} / {source['sheet']} / {column}" for source, column, _ in candidates)
        return _clarification(f"More than one numeric field matches this question: {choices}. Select one document or use the Data explorer to choose the exact sheet and column.")
    source, column, _ = candidates[0]
    frame = source["data"].reset_index(drop=True)
    values = frame[column].replace([float("inf"), -float("inf")], float("nan"))
    valid = values.dropna()
    if valid.empty:
        return _clarification(f"{source['sheet']}.{column} contains no finite numeric values to compare.")
    operation = "maximum" if high else "minimum"
    extreme = valid.max() if high else valid.min()
    positions = values.index[values.eq(extreme)].tolist()
    label_columns = {"item", "item name", "product", "product name", "name", "description", "sku", "id"}
    label = next((name for name in frame.columns if " ".join(words(name)) in label_columns), None)
    rows = []
    descriptions = []
    for position in positions[:10]:
        row = frame.iloc[position]
        row_values = {}
        for name, value in row.items():
            if pd.isna(value):
                row_values[str(name)] = None
            elif isinstance(value, (int, float)) or hasattr(value, "item"):
                scalar = value.item() if hasattr(value, "item") else value
                row_values[str(name)] = scalar if isinstance(scalar, (int, float)) and math.isfinite(scalar) else str(scalar)
            else:
                row_values[str(name)] = str(value)
        rows.append({"data_row": position + 1, "values": row_values})
        description = str(row[label]) if label is not None and pd.notna(row[label]) else f"Data row {position + 1}"
        descriptions.append(f"{description} (data row {position + 1})")
    formatted = f"{float(extreme):,.10f}".rstrip("0").rstrip(".")
    tie_note = f" {len(positions)} rows tie; showing the first 10." if len(positions) > 10 else (f" {len(positions)} rows tie." if len(positions) > 1 else "")
    answer = (
        f"1. The {operation} **{column}** is **{formatted}**: {', '.join(descriptions)}. [T1]{tie_note}\n\n"
        f"2. Compared all {len(valid):,} non-missing, finite values in **{source['source']} / {source['sheet']}** "
        f"({len(frame):,} total rows). No row filters were applied.\n\n"
        f"3. Calculated directly from the uploaded table using `{operation}({column})`, without an AI-provider call."
    )
    return {"answer": answer, "generation_status": "computed", "table_evidence": [{
        "citation_id": "T1", "source": source["source"], "sheet": source["sheet"],
        "metric": str(column), "operation": operation, "result": float(extreme),
        "rows_scanned": len(frame), "valid_values": len(valid), "ties": len(positions), "rows": rows,
    }]}
