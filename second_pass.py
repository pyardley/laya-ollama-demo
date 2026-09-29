"""Second 3-way question, asked only when the first question chose auto_reply.

The first question is the winning wording from the criteria search (a1, e3, i9),
read from the search cache. Messages it calls auto_reply are asked again, with
criteria that separate a customer how-to from a solicitation. Any other first
answer is kept.
"""

import json
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from laya import load

from tune_criteria import (
    ACTIONS,
    DEFAULT_CACHE,
    DEFAULT_MESSAGES,
    MODEL_ID,
    load_messages,
    score_rows,
)

FIRST_IDS = {"auto_reply": "a1", "escalate_to_human": "e3", "ignore": "i9"}

# Asked only after the first question has already said auto_reply.
SECOND_INSTRUCTION = "This message may be a routine how-to. What action should be taken?"
SECOND_CRITERIA = {
    "proposed": {
        "auto_reply": "The customer wants steps for their own account, order, or device.",
        "escalate_to_human": "Something is broken, charged, or compromised, and instructions will not fix it.",
        "ignore": "The sender wants us to buy, sell, rank, partner, or to enter a password, payment, or card.",
    },
    "sharper_ignore": {
        "auto_reply": "A customer how-to: password reset, pairing, tracking, firmware, settings, or store policy.",
        "escalate_to_human": "A fault, refund, cancellation, or account compromise.",
        "ignore": "SEO, a partnership, a bulk purchase, or a link asking for a password or card.",
    },
}


def load_first_pass():
    with DEFAULT_CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("_header"):
                continue
            if record["ids"] == FIRST_IDS:
                return record
    raise SystemExit("criteria_search_cache.jsonl has no a1 e3 i9 result. Run tune_criteria.py first.")


def questions_for(criteria):
    return {
        "action": {
            "instructions": SECOND_INSTRUCTION,
            "type": "choice",
            "criteria": {action: criteria[action] for action in ACTIONS},
        }
    }


def apply_second_pass(first_rows, second_by_id):
    merged = []
    for row in first_rows:
        if row["choice"] != "auto_reply":
            merged.append(dict(row))
            continue
        second = second_by_id[row["id"]]
        updated = dict(second)
        updated["label"] = row["label"]
        updated["correct"] = second["choice"] == row["label"]
        updated["p_correct"] = second["probabilities"][row["label"]]
        updated["credited_confidence"] = second["answer_confidence"] if updated["correct"] else 0.0
        updated["overridden"] = second["choice"] != "auto_reply"
        merged.append(updated)
    return merged


def accuracy(rows):
    correct = sum(1 for row in rows if row["correct"])
    return correct, len(rows)


def main():
    messages, _ = load_messages(DEFAULT_MESSAGES)
    by_id = {message["id"]: message for message in messages}
    first = load_first_pass()
    first_rows = first["rows"]
    held = [row for row in first_rows if row["choice"] == "auto_reply"]
    first_correct, n = accuracy(first_rows)
    print(f"First question a1 e3 i9: {first_correct}/{n}")
    print(f"Sent to the second question: {len(held)} messages already called auto_reply")
    print()

    print("Loading Laya...", flush=True)
    agent = load(MODEL_ID, device="cpu")
    states = [by_id[row["id"]]["text"] for row in held]

    for name, criteria in SECOND_CRITERIA.items():
        decisions = agent.predict_batch(states, questions_for(criteria), batch_size=16)
        second_rows = score_rows(
            [{"id": row["id"], "label": row["label"], "text": by_id[row["id"]]["text"]} for row in held],
            decisions,
        )
        second_by_id = {row["id"]: row for row in second_rows}
        merged = apply_second_pass(first_rows, second_by_id)
        correct, total = accuracy(merged)
        delta = correct - first_correct
        print("=" * 60)
        print(f"{name}: {correct}/{total} ({delta:+d} vs first question)")
        for action in ACTIONS:
            print(f"  {action}: {criteria[action]}")
        flips = [row for row in merged if row.get("overridden")]
        print(f"Overrides ({len(flips)}):")
        if not flips:
            print("  Second question agreed with auto_reply on every message it saw.")
        for row in merged:
            if row["id"] not in second_by_id:
                continue
            second = second_by_id[row["id"]]
            first_row = next(item for item in first_rows if item["id"] == row["id"])
            changed = "FLIP" if second["choice"] != "auto_reply" else "keep"
            was = "right" if first_row["correct"] else "wrong"
            now = "right" if row["correct"] else "wrong"
            probs = second["probabilities"]
            spread = "  ".join(f"{action[:4]} {probs[action]:.0%}" for action in ACTIONS)
            text = by_id[row["id"]]["text"]
            short = text if len(text) <= 72 else text[:69] + "..."
            print(f"  {changed:4} {row['id']} label {row['label']}  {was} -> {now}  {spread}")
            print(f"       {short}")
        print()


if __name__ == "__main__":
    main()
