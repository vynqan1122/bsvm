"""Utilities for turning local datasets into a common CSV format."""

from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SUPPORTED_EXTENSIONS = {".arff", ".csv", ".tsv", ".txt", ".data"}
DEFAULT_TARGET_NAMES = (
    "target",
    "class",
    "binaryclass",
    "label",
    "y",
    "outcome",
    "response",
)
DEFAULT_SKIP_NAMES = {
    "manifest.csv",
    "openml_datasets.csv",
    "target_overrides.csv",
}


@dataclass(frozen=True)
class LoadedTable:
    """A dataframe with lightweight source metadata."""

    frame: pd.DataFrame
    source_format: str
    target_hint: str | None = None


def _read_text_prefix(path: Path, limit: int = 4096) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            with path.open("r", encoding=encoding) as handle:
                return handle.read(limit)
        except UnicodeDecodeError:
            continue
    return ""


def looks_like_arff(path: str | Path) -> bool:
    """Return True when a file appears to be ARFF, even without extension."""

    prefix = _read_text_prefix(Path(path)).lower()
    return "@relation" in prefix and "@attribute" in prefix


def _decode_nominal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    decoded = frame.copy()
    for column in decoded.columns:
        if decoded[column].dtype != object:
            continue
        decoded[column] = decoded[column].map(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )
    return decoded


def _parse_attribute(line: str) -> tuple[str, str]:
    rest = line.strip()[len("@attribute") :].strip()
    if not rest:
        raise ValueError(f"Invalid ARFF attribute line: {line}")
    if rest[0] in {"'", '"'}:
        quote = rest[0]
        end = rest.find(quote, 1)
        if end < 0:
            raise ValueError(f"Unclosed quoted ARFF attribute: {line}")
        name = rest[1:end]
        type_text = rest[end + 1 :].strip()
    else:
        pieces = rest.split(None, 1)
        name = pieces[0]
        type_text = pieces[1].strip() if len(pieces) > 1 else ""
    return name, type_text


def _clean_arff_value(value: str) -> object:
    cleaned = value.strip()
    if cleaned == "?":
        return pd.NA
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return cleaned


def _parse_arff_rows(data_lines: list[str], n_columns: int) -> list[list[object]]:
    rows: list[list[object]] = []
    for line in data_lines:
        parsed: list[str] | None = None
        for quote in ('"', "'"):
            candidate = next(
                csv.reader([line], delimiter=",", quotechar=quote, skipinitialspace=True)
            )
            if len(candidate) == n_columns:
                parsed = candidate
                break
        if parsed is None:
            parsed = next(csv.reader([line], delimiter=",", skipinitialspace=True))
        if len(parsed) != n_columns:
            raise ValueError(
                f"ARFF row has {len(parsed)} values but {n_columns} attributes were declared"
            )
        rows.append([_clean_arff_value(value) for value in parsed])
    return rows


def _load_arff_flexible(path: Path) -> LoadedTable:
    text = path.read_text(encoding="utf-8", errors="replace")
    attributes: list[tuple[str, str]] = []
    data_lines: list[str] = []
    class_index: int | None = None
    in_data = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if line.startswith("%"):
            match = re.search(r"classindex\s*:\s*([0-9]+|last)", line, flags=re.IGNORECASE)
            if match:
                value = match.group(1).lower()
                class_index = -1 if value == "last" else int(value) - 1
            continue
        if not in_data:
            if lower.startswith("@attribute"):
                attributes.append(_parse_attribute(line))
            elif lower.startswith("@data"):
                in_data = True
            continue
        if line.startswith("%"):
            continue
        if line.startswith("{"):
            raise ValueError("Sparse ARFF rows are not supported by this converter")
        data_lines.append(line)

    if not attributes:
        raise ValueError("No ARFF attributes were found")
    if not data_lines:
        raise ValueError("No ARFF data rows were found")

    names = [name for name, _type_text in attributes]
    rows = _parse_arff_rows(data_lines, len(names))
    frame = pd.DataFrame(rows, columns=names).replace("?", pd.NA)
    for name, type_text in attributes:
        normalized_type = type_text.strip().lower()
        if normalized_type in {"numeric", "real", "integer", "int"}:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")

    target_hint = None
    if class_index is not None and -len(names) <= class_index < len(names):
        target_hint = names[class_index]
    return LoadedTable(frame=frame, source_format="arff", target_hint=target_hint)


def load_any_table(path: str | Path) -> LoadedTable:
    """Load a tabular dataset from ARFF, CSV, TSV, TXT, DATA, or XLSX."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".arff" or (suffix == "" and looks_like_arff(source)):
        return _load_arff_flexible(source)
    if suffix == ".csv":
        return LoadedTable(
            frame=pd.read_csv(source).replace("?", pd.NA),
            source_format="csv",
        )
    if suffix == ".tsv":
        return LoadedTable(
            frame=pd.read_csv(source, sep="\t").replace("?", pd.NA),
            source_format="tsv",
        )
    if suffix in {".txt", ".data"}:
        return LoadedTable(
            frame=pd.read_csv(source, sep=None, engine="python").replace("?", pd.NA),
            source_format=suffix.lstrip("."),
        )
    if suffix in {".xlsx", ".xls"}:
        return LoadedTable(
            frame=pd.read_excel(source).replace("?", pd.NA),
            source_format=suffix.lstrip("."),
        )
    raise ValueError(f"Unsupported data file format: {source.name}")


def infer_target_column(
    frame: pd.DataFrame,
    target_column: str | None = None,
    target_names: Iterable[str] = DEFAULT_TARGET_NAMES,
) -> str:
    """Infer the label column, preferring known names and then the last column."""

    if frame.empty or len(frame.columns) < 2:
        raise ValueError("A dataset must contain at least one feature and one target")
    columns = [str(column) for column in frame.columns]
    if target_column:
        if target_column in columns:
            return target_column
        lower_to_original = {column.lower(): column for column in columns}
        lowered = target_column.lower()
        if lowered in lower_to_original:
            return lower_to_original[lowered]
        raise ValueError(f"Target column {target_column!r} was not found")

    accepted = {name.lower() for name in target_names}
    for column in reversed(columns):
        normalized = re.sub(r"[^a-z0-9]+", "", column.lower())
        if normalized in accepted:
            return column
    return columns[-1]


def safe_dataset_name(path: str | Path) -> str:
    """Create a filesystem-safe dataset name from a source path."""

    source = Path(path)
    stem = source.stem if source.suffix else source.name
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return cleaned or "dataset"


def standardize_frame(
    frame: pd.DataFrame,
    *,
    target_column: str,
    target_output_name: str = "target",
) -> pd.DataFrame:
    """Move the target to the final column and rename it consistently."""

    cleaned = frame.dropna(how="all").copy()
    cleaned.columns = [str(column) for column in cleaned.columns]
    target = target_column
    if target != target_output_name and target_output_name in cleaned.columns:
        cleaned = cleaned.rename(columns={target_output_name: f"{target_output_name}_feature"})
    if target != target_output_name:
        cleaned = cleaned.rename(columns={target: target_output_name})
    feature_columns = [column for column in cleaned.columns if column != target_output_name]
    return cleaned[feature_columns + [target_output_name]]


def discover_source_files(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    skip_names: Iterable[str] = DEFAULT_SKIP_NAMES,
) -> list[Path]:
    """Find candidate raw dataset files while skipping caches and outputs."""

    root = Path(input_dir)
    skip = {name.lower() for name in skip_names}
    output_path = Path(output_dir).resolve() if output_dir is not None else None
    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        lowered_parts = {part.lower() for part in path.parts}
        if {"cache", "csv", "converted"} & lowered_parts:
            continue
        if path.name.lower() in skip:
            continue
        if output_path is not None:
            try:
                if path.resolve().is_relative_to(output_path):
                    continue
            except ValueError:
                pass
        if path.suffix.lower() in SUPPORTED_EXTENSIONS or looks_like_arff(path):
            files.append(path)
    return sorted(files, key=lambda value: value.name.lower())


def convert_file_to_csv(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    target_column: str | None = None,
    target_output_name: str = "target",
    overwrite: bool = True,
) -> dict[str, object]:
    """Convert one dataset to CSV and return a manifest row."""

    source = Path(source_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    loaded = load_any_table(source)
    inferred_target = infer_target_column(
        loaded.frame,
        target_column or loaded.target_hint,
    )
    standardized = standardize_frame(
        loaded.frame,
        target_column=inferred_target,
        target_output_name=target_output_name,
    )
    csv_path = output / f"{safe_dataset_name(source)}.csv"
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"{csv_path} already exists; pass overwrite=True")
    standardized.to_csv(csv_path, index=False)
    target_values = standardized[target_output_name].dropna()
    return {
        "dataset": csv_path.stem,
        "source_path": str(source),
        "csv_path": str(csv_path),
        "source_format": loaded.source_format,
        "rows": int(len(standardized)),
        "columns": int(len(standardized.columns)),
        "features": int(len(standardized.columns) - 1),
        "target_column_original": inferred_target,
        "target_column": target_output_name,
        "classes": int(target_values.nunique()),
        "status": "converted",
        "error": "",
    }


def load_target_overrides(path: str | Path | None) -> dict[str, str]:
    """Load optional per-dataset target choices from a small CSV file."""

    if path is None:
        return {}
    override_path = Path(path)
    if not override_path.exists():
        return {}
    table = pd.read_csv(override_path)
    if "target_column" not in table.columns:
        raise ValueError("target override CSV must contain a target_column column")
    key_columns = [
        column
        for column in ("dataset", "source_file", "source_path")
        if column in table.columns
    ]
    if not key_columns:
        raise ValueError(
            "target override CSV must contain dataset, source_file, or source_path"
        )
    overrides: dict[str, str] = {}
    for _, row in table.iterrows():
        target = str(row["target_column"]).strip()
        if not target or target.lower() == "nan":
            continue
        for key_column in key_columns:
            raw_key = str(row[key_column]).strip()
            if not raw_key or raw_key.lower() == "nan":
                continue
            path_key = Path(raw_key)
            keys = {
                raw_key.lower(),
                path_key.name.lower(),
                path_key.stem.lower(),
                safe_dataset_name(path_key).lower(),
            }
            for key in keys:
                overrides[key] = target
    return overrides


def target_override_for(source: Path, overrides: dict[str, str]) -> str | None:
    """Return a target override for a source path when one exists."""

    keys = {
        str(source).lower(),
        source.name.lower(),
        source.stem.lower(),
        safe_dataset_name(source).lower(),
    }
    for key in keys:
        if key in overrides:
            return overrides[key]
    return None


def convert_directory_to_csv(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    target_column: str | None = None,
    target_output_name: str = "target",
    recursive: bool = False,
    overwrite: bool = True,
    target_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert all discovered datasets and write a manifest CSV."""

    rows: list[dict[str, object]] = []
    overrides = target_overrides or {}
    for source in discover_source_files(
        input_dir,
        output_dir=output_dir,
        recursive=recursive,
    ):
        try:
            rows.append(
                convert_file_to_csv(
                    source,
                    output_dir,
                    target_column=target_column or target_override_for(source, overrides),
                    target_output_name=target_output_name,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by real data CLI.
            rows.append(
                {
                    "dataset": safe_dataset_name(source),
                    "source_path": str(source),
                    "csv_path": "",
                    "source_format": "",
                    "rows": 0,
                    "columns": 0,
                    "features": 0,
                    "target_column_original": target_column or "",
                    "target_column": target_output_name,
                    "classes": 0,
                    "status": "error",
                    "error": str(exc),
                }
            )
    manifest = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(Path(output_dir) / "manifest.csv", index=False)
    return manifest
