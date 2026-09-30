"""Decision rules that combine already-scored Laya calls.

None of these functions calls the model. Thresholds are fit only by
`fit_repair_thresholds`, which `fit_config.py` runs on the development set.
Applying a frozen threshold does not refit it.
"""

from common import ACTIONS, TIE_BREAK, expected_cost_action, score_predictions

# Confidence at or above this never occurs (answer_confidence is at most 1),
# so the repair rule does not fire. It is the "keep the demo choice" setting.
THRESHOLD_OFF = 1.01
THRESHOLD_GRID = [round(i / 20, 2) for i in range(21)] + [THRESHOLD_OFF]


def specialist_action(ownership, need):
    """Map the two specialist answers onto a route, or None when they do not say.

    Ownership and need are different jobs from the demo's route question. This
    map is fixed. It is not a weight fitted on the 50 messages.
    """
    if ownership["choice"] == "outsider":
        return "ignore"
    if need["choice"] == "none":
        return "ignore"
    if need["choice"] == "staff":
        return "escalate_to_human"
    if need["choice"] == "steps" and ownership["choice"] == "customer":
        return "auto_reply"
    return None


def predict_baseline(row):
    return row["demo"]["choice"]


def predict_repeat_majority(row):
    """Majority of two identical-prompt calls. A tie keeps the first call.

    Laya on CPU is expected to return the same choice twice. The second call
    is how that claim is checked. When the two choices match, this is the
    baseline and it has used two calls to get there.
    """
    first = row["demo"]["choice"]
    second = row["demo_repeat"]["choice"]
    if first == second:
        return first
    # Two different actions. Prefer the cheaper worst case, which is the
    # pre-specified tie-break, not a fitted weight.
    return min((first, second), key=lambda action: TIE_BREAK.index(action))


def predict_specialist_route(row):
    mapped = specialist_action(row["ownership"], row["need"])
    if mapped is None:
        return row["demo"]["choice"]
    return mapped


def calls_specialist_route(row):
    if specialist_action(row["ownership"], row["need"]) is None:
        return 3
    return 2


def predict_mapped_majority(row):
    """One vote from the demo, plus a vote from each specialist that names a route.

    A tie that still includes the demo choice keeps the demo. Majority therefore
    changes the demo only when both specialists outvote it.
    """
    votes = [row["demo"]["choice"]]
    if row["ownership"]["choice"] == "outsider":
        votes.append("ignore")
    if row["need"]["choice"] == "staff":
        votes.append("escalate_to_human")
    elif row["need"]["choice"] == "steps":
        votes.append("auto_reply")
    elif row["need"]["choice"] == "none":
        votes.append("ignore")
    return _majority_keep_demo(votes, row["demo"]["choice"])


def _majority_keep_demo(votes, demo_choice):
    counts = {action: 0 for action in ACTIONS}
    for vote in votes:
        counts[vote] += 1
    best = max(counts.values())
    tied = [action for action in ACTIONS if counts[action] == best]
    if demo_choice in tied:
        return demo_choice
    return min(tied, key=lambda action: TIE_BREAK.index(action))


def predict_repair(row, t_own, t_need):
    """Change the demo choice only to avoid an expensive mistake.

    Ignore to escalate when both specialists say this is the customer's own
    problem and staff have to act. Auto-reply to ignore when ownership says
    the sender is not a customer. Escalation is left alone: its worst cost is
    1, and turning it into ignore is the mistake worth 3.

    `t_own` and `t_need` are minimum answer_confidence values. The off value
    1.01 never fires, so the rule can fall back to the demo.
    """
    demo = row["demo"]["choice"]
    ownership = row["ownership"]
    need = row["need"]
    if demo == "ignore":
        if (
            ownership["choice"] == "customer"
            and ownership["answer_confidence"] >= t_own
            and need["choice"] == "staff"
            and need["answer_confidence"] >= t_need
        ):
            return "escalate_to_human"
    elif demo == "auto_reply":
        if ownership["choice"] == "outsider" and ownership["answer_confidence"] >= t_own:
            return "ignore"
    return demo


def calls_repair(row):
    """Calls a deployment of the repair rule makes, including calls that do not fire.

    The specialists are only asked when the demo choice is one the rule is
    allowed to change.
    """
    demo = row["demo"]["choice"]
    if demo == "ignore":
        return 3
    if demo == "auto_reply":
        return 2
    return 1


def repair_cost(rows, labels, cost, t_own, t_need):
    total = 0
    for row, label in zip(rows, labels):
        total += cost[label][predict_repair(row, t_own, t_need)]
    return total


def fit_repair_thresholds(rows, labels, cost, grid=None):
    """Minimise total cost. On a tie, prefer higher thresholds (fewer flips)."""
    grid = THRESHOLD_GRID if grid is None else list(grid)
    best = None
    best_key = None
    for t_own in grid:
        for t_need in grid:
            paid = repair_cost(rows, labels, cost, t_own, t_need)
            key = (paid, -t_own, -t_need)
            if best_key is None or key < best_key:
                best_key = key
                best = (t_own, t_need, paid)
    return {"t_own": best[0], "t_need": best[1], "cost": best[2]}


def loo_repair(rows, labels, cost, grid=None):
    """Leave-one-out predictions. Each message is scored with thresholds fit on the rest."""
    n = len(rows)
    predictions = []
    chosen = []
    for held in range(n):
        train_rows = [rows[i] for i in range(n) if i != held]
        train_labels = [labels[i] for i in range(n) if i != held]
        fit = fit_repair_thresholds(train_rows, train_labels, cost, grid)
        predictions.append(predict_repair(rows[held], fit["t_own"], fit["t_need"]))
        chosen.append({"t_own": fit["t_own"], "t_need": fit["t_need"], "train_cost": fit["cost"]})
    return predictions, chosen


def gate_70(row):
    """The retired 70% gate, recomputed as a control on one call. Not a candidate."""
    if row["demo"]["answer_confidence"] < 0.70:
        return "escalate_to_human"
    return row["demo"]["choice"]


def build_strategy_result(name, messages, predictions, cost, calls, **extra):
    scored = score_predictions(messages, predictions, cost, calls)
    result = {
        "strategy": name,
        "total_cost": scored["total_cost"],
        "frozen_cost": extra.pop("frozen_cost", scored["total_cost"]),
        "loo_cost": extra.pop("loo_cost", None),
        "n_correct": scored["n_correct"],
        "n_messages": scored["n_messages"],
        "accuracy": scored["accuracy"],
        "n_laya_calls": scored["n_laya_calls"],
        "confusion": scored["confusion"],
        "per_message": scored["per_message"],
        "tuned": extra.pop("tuned", False),
    }
    result.update(extra)
    if result["tuned"]:
        # The number used when ranking a tuned rule is the leave-one-out cost.
        # total_cost stays the frozen rule on these messages so the file still
        # shows what the held-out config does on the development set.
        result["comparison_cost"] = result["loo_cost"]
    else:
        result["comparison_cost"] = result["total_cost"]
    return result


def evaluate_all(bundle, cost, repair_params, with_loo=True):
    """Score every strategy. `repair_params` is the frozen full-set fit.

    `with_loo` refits the repair thresholds on each leave-one-out fold. That
    is only for the development set. A held-out file must pass with_loo=False
    so nothing is fit on it.
    """
    messages = bundle["messages"]
    rows = bundle["rows"]
    labels = [message["label"] for message in messages]
    results = []

    baseline_preds = [predict_baseline(row) for row in rows]
    results.append(build_strategy_result(
        "baseline_argmax",
        messages,
        baseline_preds,
        cost,
        [1] * len(rows),
        tuned=False,
        settings={
            "question": "demo",
            "wording": "a1 / e3 / i9",
            "rule": "highest probability",
            "temperature": "shipped",
        },
        calls_note="One Laya call per message: the tuned 3-option wording.",
        controls=_controls(messages, rows, cost),
    ))

    if rows and "demo_repeat" in rows[0]:
        repeat_preds = [predict_repeat_majority(row) for row in rows]
        identical = all(
            row["demo"]["choice"] == row["demo_repeat"]["choice"]
            and row["demo"]["probabilities"] == row["demo_repeat"]["probabilities"]
            for row in rows
        )
        results.append(build_strategy_result(
            "repeated_majority",
            messages,
            repeat_preds,
            cost,
            [2] * len(rows),
            tuned=False,
            settings={
                "question": "demo, called twice",
                "n_samples": 2,
                "temperature": "shipped",
                "identical_outputs": identical,
            },
            calls_note=(
                "Two calls of the same prompt per message. "
                + ("Outputs matched, so the vote cannot beat one call."
                   if identical else
                   "Outputs differed; the vote uses the pre-specified tie-break.")
            ),
        ))

    has_specialists = bool(rows) and "ownership" in rows[0] and "need" in rows[0]
    if has_specialists:
        results.extend(_specialist_results(messages, rows, labels, cost, repair_params, with_loo))

    _add_baseline_delta(results)
    return results


def _specialist_results(messages, rows, labels, cost, repair_params, with_loo):
    results = []
    results.append(build_strategy_result(
        "specialist_route",
        messages,
        [predict_specialist_route(row) for row in rows],
        cost,
        [calls_specialist_route(row) for row in rows],
        tuned=False,
        settings={
            "questions": ["ownership", "need", "demo as fallback"],
            "rule": "fixed map from who-is-writing and what-help-is-needed; demo only when that map abstains",
        },
        calls_note="Two specialist calls, plus the demo call only when the specialists do not name a route.",
    ))

    results.append(build_strategy_result(
        "mapped_majority",
        messages,
        [predict_mapped_majority(row) for row in rows],
        cost,
        [3] * len(rows),
        tuned=False,
        settings={
            "questions": ["demo", "ownership", "need"],
            "rule": "majority of demo plus specialist votes; a tie that includes the demo keeps the demo",
        },
        calls_note="Three Laya calls on every message.",
    ))

    hard_preds = [predict_repair(row, 0.0, 0.0) for row in rows]
    results.append(build_strategy_result(
        "costly_miss_repair",
        messages,
        hard_preds,
        cost,
        [calls_repair(row) for row in rows],
        tuned=False,
        settings={
            "questions": ["demo", "ownership", "need"],
            "t_own": 0.0,
            "t_need": 0.0,
            "rule": "flip only ignore->escalate or auto_reply->ignore, and only when the specialist choices say so",
        },
        calls_note="Demo always. Ownership and need only when the demo said ignore. Ownership only when it said auto_reply.",
    ))

    frozen_preds = [
        predict_repair(row, repair_params["t_own"], repair_params["t_need"]) for row in rows
    ]
    frozen_calls = [calls_repair(row) for row in rows]
    frozen_scored = score_predictions(messages, frozen_preds, cost, frozen_calls)
    loo_cost = None
    loo_settings = {}
    if with_loo:
        loo_preds, loo_chosen = loo_repair(rows, labels, cost)
        loo_scored = score_predictions(messages, loo_preds, cost, frozen_calls)
        loo_cost = loo_scored["total_cost"]
        loo_settings = {
            "loo_n_correct": loo_scored["n_correct"],
            "loo_confusion": loo_scored["confusion"],
            "loo_threshold_counts": _threshold_counts(loo_chosen),
        }
    # Headline predictions are the frozen rule, so a re-run with the config
    # reproduces per_message. Leave-one-out is recorded only on the development set.
    result = build_strategy_result(
        "repair_thresholds",
        messages,
        frozen_preds,
        cost,
        frozen_calls,
        tuned=True,
        frozen_cost=frozen_scored["total_cost"],
        loo_cost=loo_cost,
        settings={
            "questions": ["demo", "ownership", "need"],
            "t_own": repair_params["t_own"],
            "t_need": repair_params["t_need"],
            "grid": "0.00 to 1.00 step 0.05, plus 1.01 meaning do not fire",
            "fit": "minimise total cost on all development messages; ties take the higher thresholds",
            **loo_settings,
        },
        calls_note="Same call pattern as costly_miss_repair. Thresholds are read from config and are not refit.",
    )
    if with_loo:
        result["loo_n_correct"] = loo_settings["loo_n_correct"]
        result["loo_accuracy"] = loo_scored["accuracy"]
        result["loo_confusion"] = loo_scored["confusion"]
        result["comparison_cost"] = loo_cost
    else:
        # Held-out scoring has no leave-one-out refit. Rank this rule by the
        # cost of the frozen thresholds on the file just scored.
        result["comparison_cost"] = result["total_cost"]
    results.append(result)
    return results


def _add_baseline_delta(results):
    baseline = next(result for result in results if result["strategy"] == "baseline_argmax")
    baseline_by_id = {row["id"]: row for row in baseline["per_message"]}
    for result in results:
        flips = 0
        for row in result["per_message"]:
            previous = baseline_by_id[row["id"]]["prediction"]
            row["baseline_prediction"] = previous
            row["flipped"] = row["prediction"] != previous
            flips += int(row["flipped"])
        result["n_flips_vs_baseline"] = flips
        result["cost_delta_vs_baseline"] = result["total_cost"] - baseline["total_cost"]
        if result.get("loo_cost") is not None:
            result["loo_cost_delta_vs_baseline"] = result["loo_cost"] - baseline["total_cost"]


def _threshold_counts(chosen):
    counts = {}
    for item in chosen:
        key = f"{item['t_own']:.2f}/{item['t_need']:.2f}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _controls(messages, rows, cost):
    """Published single-call rules, scored on this run's demo probabilities.

    They are not candidates. The write-up already measured them, and a second
    question is not involved. Recomputing them checks that this run's demo
    call behaves like that write-up.
    """
    gate_preds = [gate_70(row) for row in rows]
    ec_preds = [expected_cost_action(cost, row["demo"]["probabilities"]) for row in rows]
    return {
        "confidence_gate_0.70": score_predictions(messages, gate_preds, cost, [1] * len(rows)),
        "expected_cost": score_predictions(messages, ec_preds, cost, [1] * len(rows)),
    }


def select_winner(results):
    """Lowest comparison cost, then an untuned rule, then fewer Laya calls."""
    def key(result):
        return (
            result["comparison_cost"],
            1 if result["tuned"] else 0,
            result["n_laya_calls"],
            result["strategy"],
        )
    return min(results, key=key)
