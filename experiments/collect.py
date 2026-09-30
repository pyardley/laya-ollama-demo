"""Score the fixed 3-option questions with Laya and write raw decisions.

One pass over the 50 development messages is about 20 seconds on CPU, per
question. This script asks the demo question twice (a determinism check) and
then asks the two specialist questions. It does not fit a threshold and it
does not read the labels except to carry them next to the decisions.

Example:
  python experiments/collect.py
  python experiments/collect.py --messages data/customer_messages.json
  python experiments/collect.py --messages path/to/new_messages.csv --out experiments/results/heldout/raw_decisions.json
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from common import (
    ACTIONS,
    DEFAULT_DECISIONS,
    DEFAULT_MESSAGES,
    PUBLISHED_SEARCH,
    assert_within_limits,
    audit_question,
    fingerprint,
    flatten_answers,
    is_development_set,
    load_json,
    load_messages,
    question_payload,
    write_json,
)
from questions import DEMO, NEED, OWNERSHIP, QUESTIONS

MODEL_ID = "convaiinnovations/laya"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fresh", action="store_true", help="ignore decisions already saved for this file")
    parser.add_argument(
        "--questions",
        default="all",
        help="comma-separated ids from demo,ownership,need, or 'all'. demo is always repeated once.",
    )
    return parser.parse_args()


def selected_questions(argument):
    if argument.strip() == "all":
        return list(QUESTIONS)
    by_id = {spec["id"]: spec for spec in QUESTIONS}
    chosen = []
    for name in argument.split(","):
        name = name.strip()
        if name not in by_id:
            raise SystemExit(f"Unknown question {name!r}. Choose from: {', '.join(by_id)}")
        chosen.append(by_id[name])
    if not chosen:
        raise SystemExit("Choose at least one question.")
    return chosen


def covers(payload, messages, specs, want_repeat):
    runs = payload.get("runs") or {}
    needed = [spec["id"] for spec in specs]
    if want_repeat:
        needed.append("demo_repeat")
    hashes = {message["text_sha256"] for message in messages}
    for key in needed:
        rows = runs.get(key) or []
        got = {row.get("text_sha256") for row in rows}
        if not hashes <= got:
            return False
    return True


def slice_run(rows, messages):
    by_hash = {row["text_sha256"]: row for row in rows}
    return [by_hash[message["text_sha256"]] for message in messages]


def score_questions(agent, messages, specs, batch_size):
    payload = {}
    for spec in specs:
        payload.update(question_payload(spec))
    texts = [message["text"] for message in messages]
    started = time.perf_counter()
    decisions = agent.predict_batch(texts, payload, batch_size=batch_size)
    seconds = time.perf_counter() - started
    runs = {spec["id"]: [] for spec in specs}
    for message, decision in zip(messages, decisions):
        answers = decision["answers"]
        for spec in specs:
            runs[spec["id"]].append(flatten_answers(message, answers[spec["id"]]))
    return runs, seconds


def same_answers(left, right):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a["choice"] != b["choice"] or a["probabilities"] != b["probabilities"]:
            return False
    return True


def compare_published(messages, demo_rows):
    if not PUBLISHED_SEARCH.exists():
        return {"compared": False, "reason": f"{PUBLISHED_SEARCH.name} is not in the repo"}
    published = {
        row["id"]: row
        for row in load_json(PUBLISHED_SEARCH)["best"]["messages"]
    }
    choice_ids = []
    probability_ids = []
    for message, row in zip(messages, demo_rows):
        previous = published.get(message["id"])
        if previous is None or previous.get("text") != message["text"]:
            return {
                "compared": False,
                "reason": "this file is not the 50-message development set the search was scored on",
            }
        if row["choice"] != previous["choice"]:
            choice_ids.append(message["id"])
        for action in ACTIONS:
            live = float(row["probabilities"][action])
            old = float(previous["probabilities"][action])
            if abs(live - old) > 5e-4:
                probability_ids.append(message["id"])
                break
    return {
        "compared": True,
        "choice_matches": not choice_ids,
        "probability_matches": not probability_ids,
        "n_choice_mismatches": len(choice_ids),
        "n_probability_mismatches": len(probability_ids),
        "choice_mismatch_ids": choice_ids,
        "probability_mismatch_ids": probability_ids,
    }


def collect_decisions(messages, destination, question_specs=None, include_repeat=True,
                      batch_size=8, device="cpu", fresh=False, force_eval=False):
    specs = list(question_specs) if question_specs is not None else list(QUESTIONS)
    destination = Path(destination)
    if destination.exists() and not fresh:
        existing = load_json(destination)
        if covers(existing, messages, specs, include_repeat and any(spec["id"] == "demo" for spec in specs)):
            print(f"Using decisions already in {destination}")
            return existing

    import torch
    from laya import __version__ as laya_version
    from laya import load

    torch.set_num_threads(4)
    print(f"Loading {MODEL_ID} on {device}...", flush=True)
    loaded_at = time.perf_counter()
    agent = load(MODEL_ID, device=device)
    load_seconds = time.perf_counter() - loaded_at
    training_flag = bool(agent.model.training)
    print(f"Loaded in {load_seconds:.1f}s. model.training={training_flag}", flush=True)
    if force_eval:
        agent.model.eval()
        print("Using model.eval(), matching the frozen config.", flush=True)

    audits = [audit_question(agent.tok, spec) for spec in specs]
    assert_within_limits(audits)
    for audit in audits:
        option_bits = ", ".join(
            f"{option['label']} {option['tokens']} tok" for option in audit["options"]
        )
        print(
            f"  {audit['id']}: instruction {audit['instruction_tokens']} tok, "
            f"options+masks {audit['packed_option_tokens_including_mask']}/{audit['head_token_cap']} "
            f"({option_bits})"
        )

    passes = {}
    # Match laya_demo.py and tune_criteria.py unless the frozen config had to
    # switch to eval() because the library default was not deterministic.
    inference_mode = "eval" if force_eval else "as_loaded"
    if any(spec["id"] == "demo" for spec in specs):
        print("Scoring demo, pass 1...", flush=True)
        demo_first, seconds_first = score_questions(agent, messages, [DEMO], batch_size)
        passes["demo"] = seconds_first
        print(f"  {seconds_first:.1f}s", flush=True)
        if include_repeat:
            print("Scoring demo, pass 2 (same question)...", flush=True)
            demo_second, seconds_second = score_questions(agent, messages, [DEMO], batch_size)
            passes["demo_repeat"] = seconds_second
            print(f"  {seconds_second:.1f}s", flush=True)
            identical = same_answers(demo_first["demo"], demo_second["demo"])
        else:
            demo_second = None
            identical = None
        if include_repeat and not identical:
            if inference_mode == "eval":
                raise SystemExit("Demo calls still differ under model.eval(). Refusing to score a moving target.")
            print("The two demo passes differed. Switching to model.eval() and rescoring.", flush=True)
            agent.model.eval()
            demo_first, seconds_first = score_questions(agent, messages, [DEMO], batch_size)
            demo_second, seconds_second = score_questions(agent, messages, [DEMO], batch_size)
            passes["demo"] = seconds_first
            passes["demo_repeat"] = seconds_second
            identical = same_answers(demo_first["demo"], demo_second["demo"])
            inference_mode = "eval"
            if not identical:
                raise SystemExit("Demo calls still differ after model.eval(). Refusing to score a moving target.")
        runs = {"demo": demo_first["demo"]}
        if demo_second is not None:
            runs["demo_repeat"] = demo_second["demo"]
    else:
        runs = {}
        identical = None

    others = [spec for spec in specs if spec["id"] != "demo"]
    if others:
        if inference_mode == "eval":
            agent.model.eval()
        names = ", ".join(spec["id"] for spec in others)
        print(f"Scoring {names}...", flush=True)
        other_runs, other_seconds = score_questions(agent, messages, others, batch_size)
        passes["specialists"] = other_seconds
        print(f"  {other_seconds:.1f}s", flush=True)
        runs.update(other_runs)

    published = None
    if "demo" in runs and is_development_set(messages):
        published = compare_published(messages, runs["demo"])
        if published.get("compared"):
            print(
                "Published a1/e3/i9 choices match this run: "
                f"{published['choice_matches']}. Probabilities match: {published['probability_matches']}."
            )

    shipped = {
        "temperature": list(agent.temperature),
        "temperature_by_options": dict(agent.temperature_by_options),
    }
    payload = {
        "model_id": MODEL_ID,
        "laya_version": laya_version,
        "device": device,
        "inference_mode": inference_mode,
        "model_training_after_load": training_flag,
        "load_seconds": round(load_seconds, 3),
        "pass_seconds": {key: round(value, 3) for key, value in passes.items()},
        "repeat_identical": identical,
        "published_baseline": published,
        "shipped_temperature": shipped,
        "message_fingerprint": fingerprint(messages),
        "n_messages": len(messages),
        "token_audit": audits,
        "questions": {
            spec["id"]: {
                "type": spec["type"],
                "instruction": spec["instruction"],
                "criteria": spec["criteria"],
            }
            for spec in specs
        },
        "runs": runs,
    }
    write_json(destination, payload)
    print(f"Wrote {destination}")
    return payload


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    messages = load_messages(args.messages)
    specs = selected_questions(args.questions)
    include_repeat = any(spec["id"] == "demo" for spec in specs)
    collect_decisions(
        messages,
        args.out,
        question_specs=specs,
        include_repeat=include_repeat,
        batch_size=args.batch_size,
        device=args.device,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
