"""Pearson correlation of the 30 noul columns with each other.

Reads noul_matrix.csv. Each wording is a column of 50 message scores. The
output is a 30 x 30 matrix: row i, column j is the correlation of those two
wordings across messages. A wording correlated with itself is 1.

Examples:
  python noul_correlation.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np

GROUPS = (
    ("a", "auto_reply", "a"),
    ("e", "escalate_to_human", "e"),
    ("i", "ignore", "i"),
)

ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "noul_matrix.csv"
DEFAULT_OUT = ROOT / "noul_correlation.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_columns(path):
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except FileNotFoundError:
        raise SystemExit(f"File not found: {path}")
    if len(rows) < 3:
        raise SystemExit(f"{path} needs a header and at least two message rows")
    header = rows[0]
    if header[:2] != ["message_id", "label"]:
        raise SystemExit(f"{path} must start with message_id,label")
    ids = header[2:]
    if len(ids) != 30:
        raise SystemExit(f"{path} has {len(ids)} wording columns; expected 30")
    values = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise SystemExit(f"{path} line {line_number} has {len(row)} fields; expected {len(header)}")
        try:
            values.append([float(cell) for cell in row[2:]])
        except ValueError:
            raise SystemExit(f"{path} line {line_number} has a non-numeric noul value")
    return ids, np.asarray(values, dtype=float)


def group_index(ids):
    grouped = {}
    for name, _label, prefix in GROUPS:
        grouped[name] = [index for index, qid in enumerate(ids) if qid.startswith(prefix)]
        if len(grouped[name]) != 10:
            raise SystemExit(f"expected 10 columns starting with {prefix!r}, found {len(grouped[name])}")
    return grouped


def off_diagonal_mean(corr, left, right):
    samples = []
    for i in left:
        for j in right:
            if i != j:
                samples.append(float(corr[i, j]))
    return sum(samples) / len(samples)


def print_summary(ids, corr, grouped):
    diagonal = np.diag(corr)
    print(f"Matrix {corr.shape[0]} x {corr.shape[1]}")
    print(f"Diagonal min {diagonal.min():.4f}  max {diagonal.max():.4f}")
    print("\nMean correlation, excluding a wording with itself:")
    print(f"  {'':<22}{'auto':>10}{'escalate':>10}{'ignore':>10}")
    for name, label, _prefix in GROUPS:
        cells = []
        for other, _other_label, _other_prefix in GROUPS:
            cells.append(f"{off_diagonal_mean(corr, grouped[name], grouped[other]):10.3f}")
        print(f"  {label:<22}" + "".join(cells))
    flat = corr[np.triu_indices(len(ids), k=1)]
    print(f"\nOff-diagonal  min {flat.min():.3f}  mean {flat.mean():.3f}  max {flat.max():.3f}")


def write_csv(path, ids, corr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", *ids])
        for qid, row in zip(ids, corr):
            writer.writerow([qid, *[f"{value:.4f}" for value in row]])
    print(f"\nWrote {path}")


def main():
    args = parse_args()
    ids, values = load_columns(args.matrix)
    grouped = group_index(ids)
    spread = values.std(axis=0)
    flat = [ids[index] for index, value in enumerate(spread) if value == 0]
    if flat:
        raise SystemExit(f"no variation across messages, so correlation is undefined: {', '.join(flat)}")
    corr = np.corrcoef(values, rowvar=False)
    print_summary(ids, corr, grouped)
    write_csv(args.out, ids, corr)


if __name__ == "__main__":
    main()
