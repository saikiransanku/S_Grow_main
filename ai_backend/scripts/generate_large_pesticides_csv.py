from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CSV = REPO_ROOT / "data" / "outputs" / "project_data" / "pesticides_data.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "outputs" / "project_data" / "pesticides_data_1m.csv"
DEFAULT_TARGET_ROWS = 1_000_001

FIELD_NAMES = [
    "class_label",
    "crop",
    "disease_type",
    "treatment_type",
    "recommended_pesticide",
    "active_ingredient",
    "priority",
    "effectiveness_score",
    "usage_note",
]


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_csv_key(raw_key: str | None) -> str:
    token = str(raw_key or "").replace("\ufeff", "").strip()
    return token.strip('"').strip("'").lower()


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for raw_key, value in row.items():
        key = _normalize_csv_key(raw_key)
        if not key:
            continue
        canonical[key] = str(value or "").strip()

    normalized: dict[str, str] = {}
    for key in FIELD_NAMES:
        normalized[key] = canonical.get(key, "").strip()
    return normalized


def _augment_row(seed: dict[str, str], *, row_index: int) -> dict[str, str]:
    row = dict(seed)

    base_priority = max(1, _safe_int(seed.get("priority", "3"), 3))
    row["priority"] = str(min(5, base_priority + (row_index % 2)))

    base_effectiveness = _safe_float(seed.get("effectiveness_score", "0.5"), 0.5)
    effectiveness = base_effectiveness + ((row_index % 9) - 4) * 0.01
    effectiveness = min(0.99, max(0.35, effectiveness))
    row["effectiveness_score"] = f"{effectiveness:.2f}"

    usage_note = row.get("usage_note", "").strip()
    if not usage_note:
        usage_note = "Follow local agronomy guidance and product label dosage."
    row["usage_note"] = f"{usage_note} Variant {row_index % 1000:03d}."

    return row


def generate_large_csv(source_csv: Path, output_csv: Path, target_rows: int) -> int:
    if target_rows <= 0:
        raise ValueError("target_rows must be greater than 0")
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        seeds = [_normalize_row(row) for row in csv.DictReader(handle)]
    if not seeds:
        raise ValueError(f"Source CSV has no data rows: {source_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    seed_index = 0

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
        writer.writeheader()

        while rows_written < target_rows:
            seed = seeds[seed_index]
            writer.writerow(_augment_row(seed, row_index=rows_written))
            rows_written += 1
            seed_index = (seed_index + 1) % len(seeds)

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a pesticide CSV with >1M rows for training and lookup load testing.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_CSV,
        help=f"Seed CSV path (default: {DEFAULT_SOURCE_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help=f"Number of data rows to generate (default: {DEFAULT_TARGET_ROWS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = generate_large_csv(args.source, args.output, args.rows)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
