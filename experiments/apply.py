"""Score a messages file with the frozen config. Does not fit anything.

The development set was fit by experiments/fit_config.py. This script reads
experiments/config.json, calls Laya only for the questions that config names,
and applies the saved thresholds. It will not write the config.

Score the winner only (the default):

  python experiments/apply.py --messages path/to/new_messages.csv --config experiments/config.json

Score every frozen strategy, which fills the held-out table:

  python experiments/apply.py --messages path/to/new_messages.csv --config experiments/config.json --all --out experiments/results/heldout
"""

import argparse
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    DEFAULT_MESSAGES,
    fingerprint,
    is_development_set,
    load_bundle,
    load_cost_matrix,
    load_json,
    load_messages,
    print_result,
    write_json,
)
from fit_config import _write_strategy_files
from questions import DEMO, NEED, OWNERSHIP
from rules import evaluate_all

BY_ID = {spec["id"]: spec for spec in (DEMO, OWNERSHIP, NEED)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None, help="directory for raw decisions and scores")
    parser.add_argument("--all", action="store_true", help="score every strategy in the config, not only the winner")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def questions_for(names):
    needed = set()
    repeat = False
    for name in names:
        if name == "baseline_argmax":
            needed.add("demo")
        elif name == "repeated_majority":
            needed.add("demo")
            repeat = True
        elif name in {"specialist_route", "mapped_majority", "costly_miss_repair", "repair_thresholds"}:
            needed.update(["demo", "ownership", "need"])
        else:
            raise SystemExit(f"Config names an unknown strategy {name!r}")
    return [BY_ID[key] for key in ("demo", "ownership", "need") if key in needed], repeat


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    config = load_json(args.config)
    if "repair_thresholds" not in config or "winner" not in config:
        raise SystemExit(f"{args.config} is not a frozen experiment config. Run fit_config.py on the development set.")
    messages = load_messages(args.messages)
    if is_development_set(messages) or fingerprint(messages) == config.get("development_fingerprint"):
        print("These messages are the development set. This is not a held-out result.")
    else:
        print("Scoring a new file with the frozen config. Thresholds will not be refit.")

    names = list(config["strategies"]) if args.all else [config["winner"]]
    specs, include_repeat = questions_for(names)
    # Held-out questions must be the texts that were frozen, not a later edit
    # of questions.py. Rebuild the specs from the config when it has them.
    frozen_questions = config.get("questions") or {}
    resolved = []
    for spec in specs:
        saved = frozen_questions.get(spec["id"])
        if saved:
            resolved.append({
                "id": spec["id"],
                "type": saved.get("type", "choice"),
                "instruction": saved["instruction"],
                "criteria": saved["criteria"],
            })
        else:
            resolved.append(spec)

    out_dir = args.out
    if out_dir is None:
        out_dir = Path("experiments/results/heldout")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = out_dir / "raw_decisions.json"

    from collect import collect_decisions
    collect_decisions(
        messages,
        decisions_path,
        question_specs=resolved,
        include_repeat=include_repeat,
        batch_size=args.batch_size,
        device=args.device,
        fresh=args.fresh,
        force_eval=config.get("inference_mode") == "eval",
    )
    question_keys = [spec["id"] for spec in resolved]
    if include_repeat:
        question_keys.append("demo_repeat")
    bundle = load_bundle(messages, decisions_path, questions=question_keys)
    # Prefer the frozen cost matrix over a CSV that may have changed since the fit.
    cost = config.get("cost_matrix") or load_cost_matrix()
    repair = config["repair_thresholds"]
    results = evaluate_all(bundle, cost, repair, with_loo=False)
    wanted = [result for result in results if result["strategy"] in names]
    _write_strategy_files(wanted, out_dir)
    print()
    for result in wanted:
        print_result(result)
        print()
    write_json(out_dir / "apply_manifest.json", {
        "messages": str(args.messages),
        "config": str(args.config),
        "winner_in_config": config["winner"],
        "strategies_scored": [result["strategy"] for result in wanted],
        "refit": False,
        "development_set": is_development_set(messages),
    })
    print(f"Wrote scores under {out_dir}. The config was not changed.")


if __name__ == "__main__":
    main()
