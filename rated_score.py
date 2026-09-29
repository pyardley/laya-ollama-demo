"""Cost-weighted decisions, and a second question on messages already called ignore.

Costs follow the route in laya_demo.py. Sending a real case to ignore is the
expensive mistake. An automatic reply on that same case costs less, because the
customer still gets an answer. Escalating a how-to or a piece of spam costs a
human's time and little else.

The model's probabilities are treated as P(true class). The chosen action is the
one with the lowest expected cost. The first-question wording is a1 / e3 / i9
from the criteria cache.
"""

import json
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from laya import load

from tune_criteria import ACTIONS, DEFAULT_CACHE, DEFAULT_MESSAGES, MODEL_ID, load_messages, score_rows

# cost[true_label][predicted_action]
COST = {
    "auto_reply": {"auto_reply": 0, "escalate_to_human": 1, "ignore": 2},
    "escalate_to_human": {"auto_reply": 1, "escalate_to_human": 0, "ignore": 3},
    "ignore": {"auto_reply": 2, "escalate_to_human": 1, "ignore": 0},
}

FIRST_IDS = {"auto_reply": "a1", "escalate_to_human": "e3", "ignore": "i9"}
BASELINE_IDS = {"auto_reply": "a0", "escalate_to_human": "e0", "ignore": "i0"}

IGNORE_INSTRUCTION = "This message may need no reply. What action should be taken?"
IGNORE_CRITERIA = {
    "auto_reply": "The customer wants instructions or the wording of a published policy.",
    "escalate_to_human": "A fault, a refund, a cancellation, or an account compromise. Dropping it leaves the customer stuck.",
    "ignore": "Spam, phishing, a pitch, recruitment, or text with no problem in the customer's own order, account, or device.",
}


def load_record(ids):
    with DEFAULT_CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("_header"):
                continue
            if record["ids"] == ids:
                return record
    raise SystemExit(f"Cache has no result for {ids}.")


def expected_cost(probs, action):
    return sum(probs[truth] * COST[truth][action] for truth in ACTIONS)


def rated_choice(probs):
    return min(ACTIONS, key=lambda action: (expected_cost(probs, action), action))


def realized_cost(label, choice):
    return COST[label][choice]


def summarize(rows, choice_of):
    costs = []
    correct = 0
    pairs = {(truth, pred): 0 for truth in ACTIONS for pred in ACTIONS}
    details = []
    for row in rows:
        choice = choice_of(row)
        cost = realized_cost(row["label"], choice)
        costs.append(cost)
        correct += choice == row["label"]
        pairs[(row["label"], choice)] += 1
        if choice != row["choice"] or choice != row["label"]:
            details.append((row, choice, cost))
    return {
        "correct": correct,
        "n": len(rows),
        "cost": sum(costs),
        "pairs": pairs,
        "details": details,
    }


def print_matrix():
    print("Cost when the row is the true class and the column is the action taken:")
    header = "true \\ predicted".ljust(22) + "".join(f"{action[:8]:>10}" for action in ACTIONS)
    print(header)
    for truth in ACTIONS:
        cells = "".join(f"{COST[truth][action]:10d}" for action in ACTIONS)
        print(f"{truth:<22}{cells}")
    print()


def print_summary(title, summary):
    print(
        f"{title}: accuracy {summary['correct']}/{summary['n']}  "
        f"total cost {summary['cost']}"
    )
    for truth in ACTIONS:
        for pred in ACTIONS:
            count = summary["pairs"][(truth, pred)]
            if count and truth != pred:
                print(f"  {truth} -> {pred}: {count}  (cost {COST[truth][pred]} each)")


def questions_for(criteria):
    return {
        "action": {
            "instructions": IGNORE_INSTRUCTION,
            "type": "choice",
            "criteria": {action: criteria[action] for action in ACTIONS},
        }
    }


def main():
    messages, _ = load_messages(DEFAULT_MESSAGES)
    by_id = {message["id"]: message for message in messages}
    print_matrix()

    best = load_record(FIRST_IDS)
    baseline = load_record(BASELINE_IDS)
    print_summary("Baseline wording, argmax", summarize(baseline["rows"], lambda row: row["choice"]))
    print_summary("a1 e3 i9, argmax", summarize(best["rows"], lambda row: row["choice"]))
    rated = summarize(best["rows"], lambda row: rated_choice(row["probabilities"]))
    print_summary("a1 e3 i9, lowest expected cost", rated)
    print("Decisions that move when cost replaces argmax:")
    for row, choice, cost in rated["details"]:
        if choice == row["choice"]:
            continue
        probs = row["probabilities"]
        spread = "  ".join(f"{action[:4]} {probs[action]:.0%}" for action in ACTIONS)
        was = "right" if row["correct"] else "wrong"
        now = "right" if choice == row["label"] else "wrong"
        text = by_id[row["id"]]["text"]
        short = text if len(text) <= 70 else text[:67] + "..."
        print(f"  {row['id']} {row['label']}  {row['choice']} ({was}) -> {choice} ({now})  cost {cost}  {spread}")
        print(f"    {short}")
    print()

    held = [row for row in best["rows"] if row["choice"] == "ignore"]
    print(f"Ignore re-check sees {len(held)} messages the first question called ignore.")
    print("Loading Laya...", flush=True)
    agent = load(MODEL_ID, device="cpu")
    states = [by_id[row["id"]]["text"] for row in held]
    decisions = agent.predict_batch(states, questions_for(IGNORE_CRITERIA), batch_size=16)
    second_rows = score_rows(
        [{"id": row["id"], "label": row["label"], "text": by_id[row["id"]]["text"]} for row in held],
        decisions,
    )
    second_by_id = {row["id"]: row for row in second_rows}
    merged = []
    for row in best["rows"]:
        if row["id"] not in second_by_id:
            merged.append(dict(row))
            continue
        second = second_by_id[row["id"]]
        updated = dict(second)
        updated["label"] = row["label"]
        updated["correct"] = second["choice"] == row["label"]
        merged.append(updated)

    print_summary("After ignore re-check, argmax", summarize(merged, lambda row: row["choice"]))
    print("Second question on each ignored message:")
    for row in held:
        second = second_by_id[row["id"]]
        changed = "FLIP" if second["choice"] != "ignore" else "keep"
        was = "right" if row["correct"] else "wrong"
        now = "right" if second["choice"] == row["label"] else "wrong"
        probs = second["probabilities"]
        spread = "  ".join(f"{action[:4]} {probs[action]:.0%}" for action in ACTIONS)
        text = by_id[row["id"]]["text"]
        short = text if len(text) <= 70 else text[:67] + "..."
        print(f"  {changed:4} {row['id']} label {row['label']}  {was} -> {now}  {spread}")
        print(f"       {short}")


if __name__ == "__main__":
    main()
