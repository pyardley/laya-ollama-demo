"""Shared loading, cost, and result formatting for the decision experiments.

Costs come from data/cost_matrix.csv. A Laya call, in these scripts, means one
3-option choice question scored on one message.
"""

import csv
import hashlib
import json
from pathlib import Path

ACTIONS = ("auto_reply", "escalate_to_human", "ignore")
# Lowest worst-case cost first. Escalating never costs more than 1.
TIE_BREAK = ("escalate_to_human", "auto_reply", "ignore")

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
DEFAULT_MESSAGES = ROOT / "data" / "customer_messages.json"
DEFAULT_COST = ROOT / "data" / "cost_matrix.csv"
DEFAULT_DECISIONS = EXPERIMENTS / "results" / "raw_decisions.json"
DEFAULT_CONFIG = EXPERIMENTS / "config.json"
DEFAULT_RESULTS = EXPERIMENTS / "results"
PUBLISHED_SEARCH = ROOT / "criteria_search_results.json"

OPTION_TOKEN_CAP = 48
HEAD_TOKEN_CAP = 192
# build_sequence starts cutting options once fewer than 16 head tokens remain.
HEAD_OPTION_CUT = 16


def load_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"File not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_messages(path):
    """Load labelled messages from the repo JSON shape or from a CSV.

    CSV columns are `id`, `label`, `text`. JSON is either a list of those
    objects or `{"messages": [...]}` as in data/customer_messages.json.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        return _load_csv_messages(path)
    data = load_json(path)
    if isinstance(data, dict):
        try:
            raw = data["messages"]
        except KeyError:
            raise SystemExit(f"{path} must contain a 'messages' list")
    elif isinstance(data, list):
        raw = data
    else:
        raise SystemExit(f"{path} must be a CSV, a JSON list, or a JSON object with 'messages'")
    return _clean_messages(raw, path)


def _load_csv_messages(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no header row")
        fields = [name.strip() for name in reader.fieldnames]
        missing = [name for name in ("id", "label", "text") if name not in fields]
        if missing:
            raise SystemExit(
                f"{path} must have columns id, label, text. Missing: {', '.join(missing)}"
            )
        raw = []
        for line_number, row in enumerate(reader, start=2):
            raw.append({
                "id": row.get("id"),
                "label": row.get("label"),
                "text": row.get("text"),
                "_line": line_number,
            })
    return _clean_messages(raw, path)


def _clean_messages(raw, path):
    if not raw:
        raise SystemExit(f"{path} has no messages")
    seen = set()
    cleaned = []
    for index, message in enumerate(raw, start=1):
        if not isinstance(message, dict):
            raise SystemExit(f"{path} message {index} is not an object")
        mid = str(message.get("id") or "").strip()
        label = str(message.get("label") or "").strip()
        text = str(message.get("text") or "").strip()
        where = message.get("_line", index)
        if not mid or not text:
            raise SystemExit(f"{path} row {where} needs an id and text")
        if label not in ACTIONS:
            raise SystemExit(
                f"{path} row {where} ({mid}) has label {label!r}. "
                f"Use one of: {', '.join(ACTIONS)}"
            )
        if mid in seen:
            raise SystemExit(f"{path} repeats message id {mid}")
        seen.add(mid)
        cleaned.append({"id": mid, "label": label, "text": text, "text_sha256": text_sha256(text)})
    return cleaned


def load_development_messages():
    return load_messages(DEFAULT_MESSAGES)


def same_messages(left, right):
    signature = lambda rows: [(row["id"], row["label"], row["text"]) for row in rows]
    return signature(left) == signature(right)


def is_development_set(messages):
    return same_messages(messages, load_development_messages())


def load_cost_matrix(path=DEFAULT_COST):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "true_label" not in reader.fieldnames:
            raise SystemExit(f"{path} needs a true_label column and one column per action")
        cost = {}
        for row in reader:
            truth = (row.get("true_label") or "").strip()
            if truth not in ACTIONS:
                raise SystemExit(f"{path} has true_label {truth!r}")
            cost[truth] = {}
            for action in ACTIONS:
                try:
                    cost[truth][action] = int(row[action])
                except (KeyError, TypeError, ValueError):
                    raise SystemExit(f"{path} needs an integer cost for {truth}/{action}")
    missing = [action for action in ACTIONS if action not in cost]
    if missing:
        raise SystemExit(f"{path} is missing rows for: {', '.join(missing)}")
    return cost


def realised_cost(cost, label, action):
    return cost[label][action]


def expected_cost(cost, probabilities, action):
    probs = renormalise(probabilities)
    return sum(probs[truth] * cost[truth][action] for truth in ACTIONS)


def renormalise(probabilities):
    total = sum(float(probabilities[action]) for action in ACTIONS)
    if total <= 0:
        raise ValueError("probabilities sum to 0")
    return {action: float(probabilities[action]) / total for action in ACTIONS}


def choose_min(scores, among=None):
    """Lowest score, then the earlier action in TIE_BREAK."""
    pool = list(among) if among is not None else list(ACTIONS)
    return min(pool, key=lambda action: (scores[action], TIE_BREAK.index(action)))


def expected_cost_action(cost, probabilities):
    scores = {action: expected_cost(cost, probabilities, action) for action in ACTIONS}
    return choose_min(scores)


def confusion(labels, predictions):
    matrix = {truth: {pred: 0 for pred in ACTIONS} for truth in ACTIONS}
    for truth, pred in zip(labels, predictions):
        matrix[truth][pred] += 1
    return matrix


def score_predictions(messages, predictions, cost, calls):
    if len(messages) != len(predictions) or len(messages) != len(calls):
        raise ValueError("messages, predictions, and calls must be the same length")
    labels = [message["label"] for message in messages]
    total = 0
    correct = 0
    per_message = []
    for message, prediction, n_calls in zip(messages, predictions, calls):
        paid = realised_cost(cost, message["label"], prediction)
        total += paid
        hit = prediction == message["label"]
        correct += int(hit)
        per_message.append({
            "id": message["id"],
            "label": message["label"],
            "prediction": prediction,
            "cost": paid,
            "correct": hit,
            "laya_calls": n_calls,
        })
    n = len(messages)
    return {
        "n_messages": n,
        "n_correct": correct,
        "accuracy": correct / n if n else 0.0,
        "total_cost": total,
        "n_laya_calls": sum(calls),
        "confusion": confusion(labels, predictions),
        "per_message": per_message,
    }


def format_confusion(matrix):
    header = "true \\ predicted".ljust(22) + "".join(f"{action[:18]:>20}" for action in ACTIONS)
    lines = [header]
    for truth in ACTIONS:
        cells = "".join(f"{matrix[truth][pred]:20d}" for pred in ACTIONS)
        lines.append(f"{truth:<22}{cells}")
    return "\n".join(lines)


def print_result(result):
    print(
        f"{result['strategy']}: cost {result['total_cost']}  "
        f"accuracy {result['n_correct']}/{result['n_messages']}  "
        f"Laya calls {result['n_laya_calls']}"
    )
    if result.get("loo_cost") is not None:
        print(
            f"  leave-one-out cost {result['loo_cost']}  "
            f"frozen-rule cost on these messages {result.get('frozen_cost')}"
        )
    print(format_confusion(result["confusion"]))
    note = result.get("calls_note")
    if note:
        print(note)


def question_payload(spec):
    if spec.get("type", "choice") != "choice":
        raise SystemExit(f"{spec['id']} must be a choice question with 3 options")
    criteria = spec["criteria"]
    if len(criteria) != 3:
        raise SystemExit(f"{spec['id']} has {len(criteria)} options; every call must have exactly 3")
    return {
        spec["id"]: {
            "type": "choice",
            "instructions": spec["instruction"],
            "criteria": dict(criteria),
        }
    }


def audit_question(tokenizer, spec):
    """Count tokens the way laya.common.build_sequence does, before truncation.

    A question is rejected when an option exceeds 48 tokens or when the three
    options would be cut to fit the 192-token head.
    """
    instruction = spec["instruction"]
    instruction_ids = tokenizer(
        f"choice question: {instruction}", add_special_tokens=False
    )["input_ids"]
    options = []
    packed = 0
    for label, text in spec["criteria"].items():
        rendered = " " + f"{label}: {text}"
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        kept = min(len(ids), OPTION_TOKEN_CAP)
        packed += 1 + kept  # leading MASK token plus the option
        options.append({
            "label": label,
            "tokens": len(ids),
            "cap": OPTION_TOKEN_CAP,
            "truncated": len(ids) > OPTION_TOKEN_CAP,
            "text": text,
        })
    option_budget = HEAD_TOKEN_CAP - packed
    head_cuts_options = option_budget < HEAD_OPTION_CUT
    instruction_kept = min(len(instruction_ids), max(8, option_budget))
    within = (
        not any(option["truncated"] for option in options)
        and not head_cuts_options
        and len(instruction_ids) <= max(8, option_budget)
    )
    return {
        "id": spec["id"],
        "instruction": instruction,
        "instruction_tokens": len(instruction_ids),
        "instruction_tokens_kept": instruction_kept,
        "instruction_truncated": len(instruction_ids) > max(8, option_budget),
        "options": options,
        "packed_option_tokens_including_mask": packed,
        "head_token_cap": HEAD_TOKEN_CAP,
        "tokens_left_for_instruction": option_budget,
        "head_would_cut_options": head_cuts_options,
        "within_limits": within,
    }


def assert_within_limits(audits):
    failed = [audit for audit in audits if not audit["within_limits"]]
    if not failed:
        return
    lines = ["A question would be truncated. Refusing to call Laya."]
    for audit in failed:
        lines.append(
            f"  {audit['id']}: instruction {audit['instruction_tokens']} tokens, "
            f"options+masks {audit['packed_option_tokens_including_mask']} / {HEAD_TOKEN_CAP}"
        )
        for option in audit["options"]:
            if option["truncated"]:
                lines.append(
                    f"    {option['label']}: {option['tokens']} tokens, cap {option['cap']}"
                )
        if audit["head_would_cut_options"]:
            lines.append("    the 192-token head would cut the options")
        if audit["instruction_truncated"]:
            lines.append("    the instruction would be cut")
    raise SystemExit("\n".join(lines))


def flatten_answers(message, answer):
    probabilities = {
        key: float(value) for key, value in answer["probabilities"].items()
    }
    return {
        "id": message["id"],
        "text_sha256": message["text_sha256"],
        "choice": answer["choice"],
        "answer_confidence": float(answer["answer_confidence"]),
        "probabilities": probabilities,
    }


def fingerprint(messages):
    blob = "\n".join(f"{message['id']}|{message['label']}|{message['text']}" for message in messages)
    return text_sha256(blob)


def load_bundle(messages, decisions_path, questions=None):
    payload = load_json(decisions_path)
    runs = payload.get("runs") or {}
    if questions is None:
        questions = ["demo", "ownership", "need", "demo_repeat"]
        questions = [key for key in questions if key in runs]
        if "demo" not in questions:
            questions = ["demo", "ownership", "need", "demo_repeat"]
    needed = list(questions)
    missing = [key for key in needed if key not in runs]
    if missing:
        raise SystemExit(
            f"{decisions_path} has no decisions for: {', '.join(missing)}. "
            "Run experiments/collect.py on this messages file."
        )
    aligned = {key: align_run(messages, runs[key], key) for key in needed}
    rows = []
    for index, message in enumerate(messages):
        row = {key: aligned[key][index] for key in needed}
        rows.append(row)
    return {"messages": messages, "rows": rows, "payload": payload}


def align_run(messages, rows, question_id):
    by_hash = {}
    by_id = {}
    for row in rows:
        by_id[row["id"]] = row
        by_hash[row["text_sha256"]] = row
    aligned = []
    for message in messages:
        row = by_hash.get(message["text_sha256"])
        if row is None and message["id"] in by_id and by_id[message["id"]]["text_sha256"] == message["text_sha256"]:
            row = by_id[message["id"]]
        if row is None:
            raise SystemExit(
                f"No {question_id} decision for {message['id']}. "
                "Run experiments/collect.py on this messages file."
            )
        aligned.append(row)
    return aligned
