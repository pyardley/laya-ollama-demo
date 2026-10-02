"""Run one strategy against saved Laya decisions.

The strategy scripts are thin entry points so each rule stays runnable on its
own. None of them fits a threshold. Repair thresholds are read from
experiments/config.json.
"""

import argparse
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    DEFAULT_COST,
    DEFAULT_DECISIONS,
    DEFAULT_MESSAGES,
    DEFAULT_RESULTS,
    is_development_set,
    load_bundle,
    load_cost_matrix,
    load_json,
    load_messages,
    print_result,
    write_json,
)
from rules import evaluate_all


def run(strategy_name):
    parser = argparse.ArgumentParser(
        description=f"Score the {strategy_name} rule. Reads saved Laya decisions and does not refit."
    )
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cost-matrix", type=Path, default=DEFAULT_COST)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    messages = load_messages(args.messages)
    if not args.config.exists():
        raise SystemExit(
            f"No frozen config at {args.config}. "
            "Run experiments/fit_config.py on the development set first."
        )
    config = load_json(args.config)
    repair = config["repair_thresholds"]
    development = is_development_set(messages)
    if development:
        cost = load_cost_matrix(args.cost_matrix)
    else:
        cost = config.get("cost_matrix") or load_cost_matrix(args.cost_matrix)
        print("Applying the frozen config. Thresholds are not being refit.")
    needed = {
        "baseline_argmax": ["demo"],
        "repeated_majority": ["demo", "demo_repeat"],
        "specialist_route": ["demo", "ownership", "need"],
        "mapped_majority": ["demo", "ownership", "need"],
        "costly_miss_repair": ["demo", "ownership", "need"],
        "repair_thresholds": ["demo", "ownership", "need"],
    }[strategy_name]
    bundle = load_bundle(messages, args.decisions, questions=needed)
    # Leave-one-out refits on the other 49 development messages. Never on a new file.
    results = evaluate_all(bundle, cost, repair, with_loo=development)
    result = next(item for item in results if item["strategy"] == strategy_name)
    out = args.out or (DEFAULT_RESULTS / f"{strategy_name}.json")
    write_json(out, result)
    print_result(result)
    print(f"Wrote {out}")
