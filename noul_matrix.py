"""Score every criterion wording as a yes/no against every labelled ticket.

Reads data/customer_messages.json (50 messages) and data/criteria_variants.json
(10 wordings for each of auto_reply, escalate_to_human, and ignore). Each wording
is one noul question: P(this description matches the customer message). One
batched Laya call covers all 30 questions on all 50 messages.

Writes noul_matrix.csv (message id, label, then 30 noul columns) and
noul_matrix.json (the same 50 x 30 matrix plus the question text). This script
does not call Ollama.

Examples:
  python noul_matrix.py --dry-run
  python noul_matrix.py
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

INSTRUCTION = "Does this description match the customer message?"
FALSE_TEXT = "This description does not match the customer message."
ACTIONS = ("auto_reply", "escalate_to_human", "ignore")
MODEL_ID = "convaiinnovations/laya"

ROOT = Path(__file__).resolve().parent
DEFAULT_MESSAGES = ROOT / "data" / "customer_messages.json"
DEFAULT_CRITERIA = ROOT / "data" / "criteria_variants.json"
DEFAULT_CSV = ROOT / "noul_matrix.csv"
DEFAULT_JSON = ROOT / "noul_matrix.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="check the files and print the matrix shape")
    return parser.parse_args()


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"File not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")


def load_messages(path):
    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise SystemExit(f"{path} must be an object with a 'messages' list")
    seen = set()
    cleaned = []
    for index, message in enumerate(data["messages"], start=1):
        if not isinstance(message, dict):
            raise SystemExit(f"{path} message {index} is not an object")
        mid = str(message.get("id") or "").strip()
        label = message.get("label")
        text = str(message.get("text") or "").strip()
        if not mid or not text:
            raise SystemExit(f"{path} message {index} needs an id and text")
        if label not in ACTIONS:
            raise SystemExit(f"{path} message {mid} has label {label!r}; use one of {', '.join(ACTIONS)}")
        if mid in seen:
            raise SystemExit(f"{path} repeats message id {mid}")
        seen.add(mid)
        cleaned.append({"id": mid, "label": label, "text": text})
    if not cleaned:
        raise SystemExit(f"{path} has no messages")
    return cleaned


def load_questions(path):
    data = load_json(path)
    variants = data.get("variants") if isinstance(data, dict) else None
    if not isinstance(variants, dict):
        raise SystemExit(f"{path} must contain a 'variants' object")
    questions = []
    seen = set()
    for action in ACTIONS:
        rows = variants.get(action)
        if not isinstance(rows, list) or not rows:
            raise SystemExit(f"{path} needs a non-empty variants.{action} list")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise SystemExit(f"{path} variants.{action}[{index}] is not an object")
            qid = str(row.get("id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not qid or not text:
                raise SystemExit(f"{path} variants.{action}[{index}] needs an id and text")
            if qid in seen:
                raise SystemExit(f"{path} repeats question id {qid}")
            seen.add(qid)
            questions.append({"id": qid, "action": action, "text": text})
    return questions


def noul_questions(questions):
    return {
        question["id"]: {
            "type": "noul",
            "instructions": INSTRUCTION,
            "criteria": {"true": question["text"], "false": FALSE_TEXT},
        }
        for question in questions
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def print_summary(messages, questions, matrix):
    print(f"\nMatrix {len(messages)} x {len(questions)}")
    flat = [value for row in matrix for value in row]
    print(f"noul  min {min(flat):.4f}  mean {mean(flat):.4f}  max {max(flat):.4f}")
    print("\nMean noul of each wording on messages of each true label:")
    header = f"  {'id':<4} {'wording for':<20}" + "".join(f"{action[:8]:>10}" for action in ACTIONS)
    print(header)
    for column, question in enumerate(questions):
        cells = []
        for action in ACTIONS:
            values = [matrix[row][column] for row, message in enumerate(messages) if message["label"] == action]
            cells.append(f"{mean(values):10.3f}")
        print(f"  {question['id']:<4} {question['action']:<20}" + "".join(cells))


def write_outputs(csv_path, json_path, messages, questions, matrix, seconds):
    question_ids = [question["id"] for question in questions]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["message_id", "label", *question_ids])
        for message, row in zip(messages, matrix):
            writer.writerow([message["id"], message["label"], *[f"{value:.4f}" for value in row]])

    payload = {
        "model": MODEL_ID,
        "instruction": INSTRUCTION,
        "false_text": FALSE_TEXT,
        "noul_meaning": "P(the wording matches the customer message). Row is a message, column is a wording.",
        "seconds": round(seconds, 1),
        "shape": [len(messages), len(questions)],
        "question_ids": question_ids,
        "questions": questions,
        "message_ids": [message["id"] for message in messages],
        "labels": [message["label"] for message in messages],
        "matrix": matrix,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    messages = load_messages(args.messages)
    questions = load_questions(args.criteria)
    print(f"{len(messages)} messages x {len(questions)} noul questions")
    print(f"Instruction: {INSTRUCTION}")
    if args.dry_run:
        print("Dry run: files loaded, model not called.")
        return

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    print("Loading Laya...", flush=True)
    started = time.perf_counter()
    from laya import load
    agent = load(MODEL_ID, device=args.device)
    print(f"Loaded in {time.perf_counter() - started:.1f}s", flush=True)

    print(f"Scoring {len(messages) * len(questions)} noul values...", flush=True)
    scored_at = time.perf_counter()
    decisions = agent.predict_batch(
        [message["text"] for message in messages],
        noul_questions(questions),
        batch_size=args.batch_size,
    )
    seconds = time.perf_counter() - scored_at
    print(f"Scored in {seconds:.1f}s", flush=True)

    matrix = []
    for decision in decisions:
        answers = decision["answers"]
        matrix.append([float(answers[question["id"]]["noul"]) for question in questions])

    print_summary(messages, questions, matrix)
    write_outputs(args.csv, args.json, messages, questions, matrix, seconds)


if __name__ == "__main__":
    main()
