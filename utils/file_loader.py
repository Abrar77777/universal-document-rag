import json
from pathlib import Path

import docx
import pandas as pd
from pypdf import PdfReader


def table_text(df: pd.DataFrame, sheet_name: str) -> str:
    lines = []
    for number, row in enumerate(df.itertuples(index=False, name=None), 1):
        fields = "; ".join(f"{column}: {value}" for column, value in zip(df.columns, row) if pd.notna(value))
        if fields:
            lines.append(f"Sheet {sheet_name}, row {number}: {fields}")
    return "\n".join(lines)

def load_file(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text

    elif suffix in {".txt", ".md"}:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif suffix == ".docx":
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        table_rows = []
        for table in doc.tables:
            for row in table.rows:
                table_rows.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(paragraphs + table_rows)

    elif suffix == ".csv":
        return table_text(pd.read_csv(file_path), "data")

    elif suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(file_path, sheet_name=None)
        parts = []
        for sheet_name, df in sheets.items():
            parts.append(f"Sheet: {sheet_name}")
            parts.append(table_text(df, sheet_name))
        return "\n".join(parts)

    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=True)

    else:
        raise ValueError("Unsupported file type")

