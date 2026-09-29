"""Search Laya criteria wording for the highest confidence on the correct action.

Customer messages and candidate wordings are written ahead of time (see data/).
This script only reads those files and scores them with Laya. It does not call Ollama.

Each message contributes its answer_confidence when the chosen action matches the
label, and 0 when it does not. The winning combination is the one with the highest
mean of that credited confidence. Accuracy, then mean probability on the label,
then the unweighted mean across the three actions, break ties.

A full grid of 10 wordings per action is 1,000 combinations. One combination
is a single batched Laya call over every message (about 20 seconds on CPU),
so the grid takes a few hours. --strategy coordinate tries every wording while
optimising one action at a time, which is a few rounds of about 30 calls.
Results are written as they go, and a JSONL cache lets a stopped run resume.

Examples:
  python tune_criteria.py --dry-run
  python tune_criteria.py --only-baseline
  python tune_criteria.py
  python tune_criteria.py --strategy coordinate
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

# Same question and starting wordings as laya_demo.py. The search compares against these.
INSTRUCTIONS = "What action should be taken for this customer support ticket?"
ACTIONS = ("auto_reply", "escalate_to_human", "ignore")
BASELINE = {
    "auto_reply": "The issue is simple and can be addressed automatically.",
    "escalate_to_human": "The hardware issue or replacement request needs a human agent.",
    "ignore": "The message is spam or completely irrelevant.",
}
# Reported beside the search score. laya_demo.py withholds every route below this.
CONFIDENCE_THRESHOLD = 0.70
MODEL_ID = "convaiinnovations/laya"
CACHE_VERSION = 1

ROOT = Path(__file__).resolve().parent
DEFAULT_MESSAGES = ROOT / "data" / "customer_messages.json"
DEFAULT_CRITERIA = ROOT / "data" / "criteria_variants.json"
DEFAULT_OUT = ROOT / "criteria_search_results.json"
DEFAULT_CACHE = ROOT / "criteria_search_cache.jsonl"

OBJECTIVE = (
    "Mean credited answer_confidence: answer_confidence when the chosen action "
    "matches the labeled action, otherwise 0."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--strategy", choices=("grid", "coordinate"), default="grid")
    parser.add_argument("--rounds", type=int, default=3, help="coordinate ascent passes (default 3)")
    parser.add_argument("--only-baseline", action="store_true", help="score the laya_demo.py wording and stop")
    parser.add_argument("--max-combos", type=int, default=0, help="stop after this many new model calls (0 = no cap)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--fresh", action="store_true", help="ignore and replace the resume cache")
    parser.add_argument("--dry-run", action="store_true", help="validate the JSON and print the grid size")
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
    policy = {}
    if isinstance(data, dict):
        policy = data.get("policy") or {}
        try:
            messages = data["messages"]
        except KeyError:
            raise SystemExit(f"{path} must contain a 'messages' list")
    elif isinstance(data, list):
        messages = data
    else:
        raise SystemExit(f"{path} must be a list of messages or an object with 'messages'")
    if not messages:
        raise SystemExit(f"{path} has no messages")

    seen = set()
    cleaned = []
    for index, message in enumerate(messages, start=1):
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
    return cleaned, policy


def load_variants(path):
    data = load_json(path)
    if isinstance(data, dict) and "variants" in data:
        raw = data["variants"]
    else:
        raw = data
    if not isinstance(raw, dict) or set(raw) != set(ACTIONS):
        raise SystemExit(f"{path} variants must have exactly these keys: {', '.join(ACTIONS)}")

    variants = {}
    for action in ACTIONS:
        items = raw[action]
        if not isinstance(items, list) or not items:
            raise SystemExit(f"{path} {action} needs a non-empty list of wordings")
        seen_ids = set()
        seen_text = set()
        cleaned = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise SystemExit(f"{path} {action} item {index} is not an object")
            vid = str(item.get("id") or "").strip()
            text = str(item.get("text") or "").strip()
            if not vid or not text:
                raise SystemExit(f"{path} {action} item {index} needs an id and text")
            if vid in seen_ids:
                raise SystemExit(f"{path} repeats {action} id {vid}")
            seen_ids.add(vid)
            if text in seen_text:
                print(f"Warning: duplicate {action} wording ({vid}); matching combinations are scored once.")
            seen_text.add(text)
            cleaned.append({"id": vid, "text": text})
        variants[action] = cleaned
    return variants


def fingerprint(messages, variants):
    payload = {
        "cache_version": CACHE_VERSION,
        "instructions": INSTRUCTIONS,
        "baseline": BASELINE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "messages": messages,
        "variants": variants,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def combo_texts(combo):
    return tuple(combo[action]["text"] for action in ACTIONS)


def is_baseline(record):
    return all(record["texts"][action] == BASELINE[action] for action in ACTIONS)


def baseline_combo(variants):
    combo = {}
    for action in ACTIONS:
        match = next((item for item in variants[action] if item["text"] == BASELINE[action]), None)
        combo[action] = match or {"id": "baseline", "text": BASELINE[action]}
    return combo


def grid_combos(variants):
    """Baseline first, then every other wording combination, skipping duplicate text."""
    combos = [baseline_combo(variants)]
    seen = {combo_texts(combos[0])}
    lists = [variants[action] for action in ACTIONS]
    for picks in itertools.product(*lists):
        combo = {action: item for action, item in zip(ACTIONS, picks)}
        key = combo_texts(combo)
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)
    return combos


def questions_for(combo):
    return {
        "action": {
            "instructions": INSTRUCTIONS,
            "type": "choice",
            "criteria": {action: combo[action]["text"] for action in ACTIONS},
        }
    }


def summarize(rows):
    n = len(rows)
    if n == 0:
        raise ValueError("summarize() needs at least one row")
    n_correct = sum(1 for row in rows if row["correct"])
    n_actionable = sum(
        1 for row in rows if row["correct"] and row["answer_confidence"] >= CONFIDENCE_THRESHOLD
    )
    by_label = {}
    for action in ACTIONS:
        subset = [row for row in rows if row["label"] == action]
        n_label = len(subset)
        correct_rows = [row for row in subset if row["correct"]]
        by_label[action] = {
            "n": n_label,
            "n_correct": len(correct_rows),
            "accuracy": (len(correct_rows) / n_label) if n_label else 0.0,
            "mean_credited_confidence": (
                sum(row["credited_confidence"] for row in subset) / n_label if n_label else 0.0
            ),
            "mean_p_correct": (sum(row["p_correct"] for row in subset) / n_label) if n_label else 0.0,
            "mean_answer_confidence_when_correct": (
                sum(row["answer_confidence"] for row in correct_rows) / len(correct_rows)
                if correct_rows else None
            ),
        }
    class_means = [by_label[action]["mean_credited_confidence"] for action in ACTIONS if by_label[action]["n"]]
    return {
        "n": n,
        "n_correct": n_correct,
        "mean_credited_confidence": sum(row["credited_confidence"] for row in rows) / n,
        "accuracy": n_correct / n,
        "mean_p_correct": sum(row["p_correct"] for row in rows) / n,
        "macro_credited_confidence": sum(class_means) / len(class_means) if class_means else 0.0,
        "n_correct_above_threshold": n_actionable,
        "fraction_correct_above_threshold": n_actionable / n,
        "by_label": by_label,
    }


def rank_key(summary):
    return (
        summary["mean_credited_confidence"],
        summary["accuracy"],
        summary["mean_p_correct"],
        summary["macro_credited_confidence"],
    )


def format_ids(record):
    return " ".join(record["ids"][action] for action in ACTIONS)


def class_counts(summary):
    parts = []
    for action in ACTIONS:
        block = summary["by_label"][action]
        parts.append(f"{action} {block['n_correct']}/{block['n']}")
    return ", ".join(parts)


def score_rows(messages, decisions):
    rows = []
    for message, decision in zip(messages, decisions):
        action = decision["answers"]["action"]
        expected = message["label"]
        choice = action["choice"]
        correct = choice == expected
        answer_confidence = float(action["answer_confidence"])
        probabilities = {key: float(value) for key, value in action["probabilities"].items()}
        rows.append({
            "id": message["id"],
            "label": expected,
            "choice": choice,
            "correct": correct,
            "answer_confidence": answer_confidence,
            "credited_confidence": answer_confidence if correct else 0.0,
            "p_correct": probabilities[expected],
            "probabilities": probabilities,
        })
    return rows


def warn_long_options(agent, variants):
    """Laya keeps the first 48 tokens of each rendered option (`label: wording`)."""
    tok = agent.tok
    for action, items in variants.items():
        for item in items:
            rendered = f" {action}: {item['text']}"
            n_tokens = len(tok(rendered, add_special_tokens=False)["input_ids"])
            if n_tokens > 48:
                print(f"Warning: {item['id']} renders to {n_tokens} tokens and will be cut to 48.")


def load_cache(path, expected_fingerprint, fresh):
    if fresh and path.exists():
        path.unlink()
    evaluated = {}
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"_header": True, "fingerprint": expected_fingerprint}) + "\n")
        return evaluated

    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise SystemExit(f"{path} is empty. Delete it or pass --fresh.")
    header = json.loads(lines[0])
    if not header.get("_header") or header.get("fingerprint") != expected_fingerprint:
        raise SystemExit(
            f"{path} was built for different messages, criteria, or scoring. "
            "Pass --fresh to start again."
        )
    for line in lines[1:]:
        record = json.loads(line)
        evaluated[tuple(record["texts"][action] for action in ACTIONS)] = record
    return evaluated


def append_cache(handle, record):
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def rows_with_text(rows, messages_by_id):
    detailed = []
    for row in rows:
        item = dict(row)
        item["text"] = messages_by_id[row["id"]]["text"]
        detailed.append(item)
    return detailed


def compact_summary(record):
    summary = record["summary"]
    return {
        "ids": record["ids"],
        "baseline": is_baseline(record),
        "mean_credited_confidence": summary["mean_credited_confidence"],
        "accuracy": summary["accuracy"],
        "n_correct": summary["n_correct"],
        "mean_p_correct": summary["mean_p_correct"],
        "macro_credited_confidence": summary["macro_credited_confidence"],
        "n_correct_above_threshold": summary["n_correct_above_threshold"],
        "by_label": {
            action: {
                "n": summary["by_label"][action]["n"],
                "n_correct": summary["by_label"][action]["n_correct"],
                "mean_credited_confidence": summary["by_label"][action]["mean_credited_confidence"],
            }
            for action in ACTIONS
        },
    }


def write_results(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_payload(messages, evaluated, finished, strategy, model_seconds):
    ranked = sorted(evaluated.values(), key=lambda record: rank_key(record["summary"]), reverse=True)
    best = ranked[0]
    baseline = next((record for record in ranked if is_baseline(record)), None)
    messages_by_id = {message["id"]: message for message in messages}

    def detail(record):
        if record is None:
            return None
        body = compact_summary(record)
        body["texts"] = record["texts"]
        body["messages"] = rows_with_text(record["rows"], messages_by_id)
        return body

    return {
        "objective": OBJECTIVE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "strategy": strategy,
        "finished": finished,
        "model_seconds": round(model_seconds, 3),
        "n_evaluated": len(evaluated),
        "n_messages": len(messages),
        "best": detail(best),
        "baseline": detail(baseline),
        "leaderboard": [compact_summary(record) for record in ranked],
    }


def print_banner(messages, variants, policy):
    counts = {action: sum(1 for message in messages if message["label"] == action) for action in ACTIONS}
    count_text = ", ".join(f"{action} {counts[action]}" for action in ACTIONS)
    sizes = " x ".join(str(len(variants[action])) for action in ACTIONS)
    print(f"Messages: {len(messages)} ({count_text})")
    print(f"Wordings: {sizes} = {len(grid_combos(variants))} combinations")
    print(f"Objective: {OBJECTIVE}")
    if policy:
        print("Labels:")
        for action in ACTIONS:
            if action in policy:
                print(f"  {action}: {policy[action]}")
    print()


def print_record(title, record):
    summary = record["summary"]
    print(title)
    print(f"  credited confidence : {summary['mean_credited_confidence']:.3f}")
    print(f"  accuracy            : {summary['n_correct']}/{summary['n']} ({summary['accuracy']:.1%})")
    print(f"  mean P(label)       : {summary['mean_p_correct']:.3f}")
    print(f"  macro credited      : {summary['macro_credited_confidence']:.3f}")
    print(
        f"  correct and >= {CONFIDENCE_THRESHOLD:.0%} : "
        f"{summary['n_correct_above_threshold']}/{summary['n']}"
    )
    for action in ACTIONS:
        block = summary["by_label"][action]
        conditional = block["mean_answer_confidence_when_correct"]
        conditional_text = f"{conditional:.3f}" if conditional is not None else "n/a"
        print(
            f"  {action:<20} {block['n_correct']:>2}/{block['n']:<2}  "
            f"credited {block['mean_credited_confidence']:.3f}  "
            f"P(label) {block['mean_p_correct']:.3f}  "
            f"conf|correct {conditional_text}"
        )


def print_criteria(record):
    for action in ACTIONS:
        print(f"  {action} [{record['ids'][action]}]")
        print(f"    {record['texts'][action]}")


def print_misses(record, messages_by_id):
    misses = [row for row in record["rows"] if not row["correct"]]
    print(f"Misses ({len(misses)}):")
    if not misses:
        print("  Every labeled message was chosen correctly.")
        return
    for row in misses:
        text = messages_by_id[row["id"]]["text"]
        short = text if len(text) <= 78 else text[:75] + "..."
        spread = "  ".join(f"{action} {row['probabilities'][action]:.0%}" for action in ACTIONS)
        print(f"  {row['id']}  {row['label']} -> {row['choice']}")
        print(f"    {spread}")
        print(f"    {short}")


def print_leaderboard(records, top):
    shown = records[:top]
    print(f"Top {len(shown)}:")
    print(f"  {'rank':<5} {'auto':<8} {'esc':<8} {'ign':<8} {'credited':>9} {'accuracy':>9} {'P(label)':>9} {'>=70%':>6}")
    for rank, record in enumerate(shown, start=1):
        summary = record["summary"]
        ids = record["ids"]
        mark = "*" if is_baseline(record) else " "
        print(
            f"  {rank:<4}{mark} {ids['auto_reply']:<8} {ids['escalate_to_human']:<8} {ids['ignore']:<8} "
            f"{summary['mean_credited_confidence']:9.3f} {summary['accuracy']:9.1%} "
            f"{summary['mean_p_correct']:9.3f} {summary['n_correct_above_threshold']:6d}"
        )
    baseline_rank = next((index for index, record in enumerate(records, start=1) if is_baseline(record)), None)
    if any(is_baseline(record) for record in shown):
        print("  * current laya_demo.py wording")
    elif baseline_rank is not None:
        baseline = records[baseline_rank - 1]["summary"]
        print(
            f"  Current laya_demo.py wording is rank {baseline_rank}: "
            f"credited {baseline['mean_credited_confidence']:.3f}, accuracy {baseline['accuracy']:.1%}"
        )


def print_report(evaluated, messages, top):
    ranked = sorted(evaluated.values(), key=lambda record: rank_key(record["summary"]), reverse=True)
    best = ranked[0]
    baseline = next((record for record in ranked if is_baseline(record)), None)
    messages_by_id = {message["id"]: message for message in messages}

    print()
    print(f"Evaluated {len(ranked)} combinations.")
    print("=" * 60)
    only_current = len(ranked) == 1 and is_baseline(best)
    print("CURRENT WORDING" if only_current else "BEST CRITERIA")
    print("=" * 60)
    print_record(format_ids(best), best)
    print()
    print_criteria(best)
    print()
    print_misses(best, messages_by_id)

    if baseline is not None and baseline is not best:
        base = baseline["summary"]["mean_credited_confidence"]
        gain = best["summary"]["mean_credited_confidence"] - base
        print()
        print("=" * 60)
        print("VERSUS CURRENT WORDING")
        print("=" * 60)
        print_record(format_ids(baseline), baseline)
        print()
        print(f"Credited confidence {base:.3f} -> {best['summary']['mean_credited_confidence']:.3f} ({gain:+.3f})")
        print(
            f"Accuracy {baseline['summary']['accuracy']:.1%} -> {best['summary']['accuracy']:.1%}"
        )
        print("Paste into laya_demo.py:")
        print(json.dumps({action: best["texts"][action] for action in ACTIONS}, indent=4))
    elif baseline is not None and len(ranked) > 1:
        print()
        print("The current laya_demo.py wording has the best score of the combinations evaluated.")

    print()
    print_leaderboard(ranked, top)


def eta_text(done, total, model_seconds, model_calls):
    if not total or model_calls == 0 or done >= total:
        return ""
    remaining = (total - done) * (model_seconds / model_calls)
    minutes, seconds = divmod(int(remaining), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"  eta {hours}h{minutes:02d}m"
    return f"  eta {minutes}m{seconds:02d}s"


def make_evaluator(agent, messages, batch_size, cache_handle, evaluated, progress):
    states = [message["text"] for message in messages]

    def evaluate(combo):
        key = combo_texts(combo)
        cached = key in evaluated
        if not cached:
            if progress["max_combos"] and progress["model_calls"] >= progress["max_combos"]:
                return None
            started = time.perf_counter()
            decisions = agent.predict_batch(states, questions_for(combo), batch_size=batch_size)
            elapsed = time.perf_counter() - started
            progress["model_seconds"] += elapsed
            progress["model_calls"] += 1
            rows = score_rows(messages, decisions)
            record = {
                "ids": {action: combo[action]["id"] for action in ACTIONS},
                "texts": {action: combo[action]["text"] for action in ACTIONS},
                "summary": summarize(rows),
                "rows": rows,
            }
            evaluated[key] = record
            append_cache(cache_handle, record)
            progress["since_save"] += 1
        record = evaluated[key]
        summary = record["summary"]
        improved = rank_key(summary) > progress["best_rank"]
        if improved:
            progress["best_rank"] = rank_key(summary)
        if not cached or improved:
            total = progress["total"]
            done = progress["done_fn"]()
            total_text = f"/{total}" if total else ""
            note = "  new best" if improved else ""
            print(
                f"[{done}{total_text}] {format_ids(record):<16} "
                f"credited {summary['mean_credited_confidence']:.3f}  "
                f"acc {summary['accuracy']:.1%}  "
                f"P(label) {summary['mean_p_correct']:.3f}"
                f"{note}{eta_text(done, total, progress['model_seconds'], progress['model_calls'])}",
                flush=True,
            )
            if improved:
                print(f"         {class_counts(summary)}", flush=True)
        return record

    return evaluate


def run_grid(combos, evaluate, progress):
    progress["total"] = len(combos)
    progress["done_fn"] = lambda: sum(1 for combo in combos if combo_texts(combo) in progress["evaluated"])
    for combo in combos:
        if evaluate(combo) is None:
            print(f"Stopped at --max-combos {progress['max_combos']}.")
            return False
    return True


def run_coordinate(variants, start, evaluate, rounds, progress):
    current = start
    # The starting point is part of the search, so score it before the axes move.
    if evaluate(current) is None:
        return False
    progress["total"] = None
    progress["done_fn"] = lambda: len(progress["evaluated"])
    for round_index in range(1, rounds + 1):
        print(f"Coordinate round {round_index}/{rounds}", flush=True)
        changed = False
        for action in ACTIONS:
            best_combo = current
            best_rank = rank_key(evaluate(current)["summary"])
            for variant in variants[action]:
                trial = dict(current)
                trial[action] = variant
                record = evaluate(trial)
                if record is None:
                    print(f"Stopped at --max-combos {progress['max_combos']}.")
                    return False
                trial_rank = rank_key(record["summary"])
                if trial_rank > best_rank:
                    best_combo = trial
                    best_rank = trial_rank
            if combo_texts(best_combo) != combo_texts(current):
                changed = True
                current = best_combo
        if not changed:
            print(f"Coordinate search settled on round {round_index}.")
            return True
    return True


def install_periodic_save(evaluate, progress, save):
    """Save the leaderboard every N new model calls without hiding evaluate()."""

    def wrapped(combo):
        record = evaluate(combo)
        if progress["save_every"] > 0 and progress["since_save"] >= progress["save_every"]:
            progress["since_save"] = 0
            save(False)
        return record

    return wrapped


def main():
    args = parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    messages, policy = load_messages(args.messages)
    variants = load_variants(args.criteria)
    print_banner(messages, variants, policy)
    if args.dry_run:
        print("Dry run: files loaded, model not called.")
        return

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    print("Loading Laya...", flush=True)
    started = time.perf_counter()
    from laya import load
    agent = load(MODEL_ID, device=args.device)
    print(f"Loaded in {time.perf_counter() - started:.1f}s", flush=True)
    warn_long_options(agent, variants)

    token = fingerprint(messages, variants)
    evaluated = load_cache(args.cache, token, args.fresh)
    if evaluated:
        print(f"Resumed {len(evaluated)} cached combinations.", flush=True)

    if args.only_baseline:
        strategy = "baseline"
        plan = [baseline_combo(variants)]
    elif args.strategy == "grid":
        strategy = "grid"
        plan = grid_combos(variants)
    else:
        strategy = "coordinate"
        plan = None

    progress = {
        "evaluated": evaluated,
        "model_seconds": 0.0,
        "model_calls": 0,
        "max_combos": args.max_combos,
        "since_save": 0,
        "save_every": args.save_every,
        "best_rank": (-1.0, -1.0, -1.0, -1.0),
        "total": len(plan) if plan is not None else None,
        "done_fn": (lambda: sum(1 for combo in plan if combo_texts(combo) in evaluated))
        if plan is not None else (lambda: len(evaluated)),
    }
    if evaluated:
        progress["best_rank"] = max(rank_key(record["summary"]) for record in evaluated.values())

    def save(done):
        if not evaluated:
            return
        payload = build_payload(messages, evaluated, done, strategy, progress["model_seconds"])
        write_results(args.out, payload)

    cache_handle = args.cache.open("a", encoding="utf-8")
    finished = False
    try:
        evaluate = make_evaluator(
            agent, messages, args.batch_size, cache_handle, evaluated, progress
        )
        evaluate = install_periodic_save(evaluate, progress, save)
        if strategy == "coordinate":
            finished = run_coordinate(
                variants, baseline_combo(variants), evaluate, args.rounds, progress
            )
        else:
            finished = run_grid(plan, evaluate, progress)
        save(finished)
        if evaluated:
            print_report(evaluated, messages, args.top)
            print(f"\nWrote {args.out}")
        calls = progress["model_calls"]
        print(f"Model time {progress['model_seconds']:.1f}s across {calls} new combinations.")
    except KeyboardInterrupt:
        print("\nStopped early. Partial results saved.", flush=True)
        save(False)
        if evaluated:
            print(f"Wrote {args.out}")
    finally:
        cache_handle.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stdout = None
        sys.exit(0)
