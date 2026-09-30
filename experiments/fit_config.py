"""Fit repair thresholds on the development set and freeze them.

This script refuses any file other than the original 50 messages. A later
file is scored with experiments/apply.py, which reads the written config and
does not fit anything.

The frozen thresholds are the ones that minimise cost on all 50. Leave-one-out
cost is recorded beside them and is what the ranking uses. The held-out run
must not call this script.
"""

import argparse
import csv
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    DEFAULT_COST,
    DEFAULT_DECISIONS,
    DEFAULT_RESULTS,
    fingerprint,
    is_development_set,
    load_bundle,
    load_cost_matrix,
    load_development_messages,
    load_messages,
    write_json,
)
from rules import evaluate_all, fit_repair_thresholds, select_winner

STRATEGY_ORDER = (
    "baseline_argmax",
    "repeated_majority",
    "specialist_route",
    "mapped_majority",
    "costly_miss_repair",
    "repair_thresholds",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--messages", type=Path, default=None)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--cost-matrix", type=Path, default=DEFAULT_COST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def fit_and_write(messages, decisions_path, cost_path, config_path, results_dir):
    if not is_development_set(messages):
        raise SystemExit(
            "fit_config.py only fits the original 50 messages in data/customer_messages.json. "
            "Score a new file with experiments/apply.py, which reads experiments/config.json "
            "and does not refit."
        )
    bundle = load_bundle(
        messages, decisions_path, questions=["demo", "demo_repeat", "ownership", "need"]
    )
    cost = load_cost_matrix(cost_path)
    labels = [message["label"] for message in messages]
    fit = fit_repair_thresholds(bundle["rows"], labels, cost)
    results = evaluate_all(bundle, cost, fit, with_loo=True)
    by_name = {result["strategy"]: result for result in results}
    winner = select_winner(results)
    payload = bundle["payload"]
    config = {
        "schema_version": 1,
        "fitted_on": "data/customer_messages.json",
        "n_messages": len(messages),
        "development_fingerprint": fingerprint(load_development_messages()),
        "do_not_refit": True,
        "model_id": payload.get("model_id"),
        "laya_version": payload.get("laya_version"),
        "device": payload.get("device"),
        "inference_mode": payload.get("inference_mode"),
        "repeat_identical": payload.get("repeat_identical"),
        "published_baseline": payload.get("published_baseline"),
        "cost_matrix": cost,
        "cost_matrix_path": str(cost_path),
        "questions": payload.get("questions"),
        "token_audit": payload.get("token_audit"),
        "repair_thresholds": {
            "t_own": fit["t_own"],
            "t_need": fit["t_need"],
            "resubstitution_cost": fit["cost"],
            "grid": "0.00 to 1.00 step 0.05, plus 1.01 meaning do not fire",
            "tie_break": "lowest cost, then higher t_own, then higher t_need",
            "fit_on": "all 50 development messages",
        },
        "winner": winner["strategy"],
        "winner_comparison_cost": winner["comparison_cost"],
        "winner_selection": (
            "Lowest development cost. Tuned repair_thresholds is ranked on its "
            "leave-one-out cost, not on the cost of the frozen thresholds reapplied "
            "to the same 50. Ties prefer an untuned rule, then fewer Laya calls."
        ),
        "strategies": {
            name: _strategy_summary(by_name[name])
            for name in STRATEGY_ORDER
            if name in by_name
        },
    }
    write_json(config_path, config)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_strategy_files(results, results_dir)
    print(f"Wrote {config_path}")
    print(f"Winner: {winner['strategy']}  comparison cost {winner['comparison_cost']}")
    print(
        f"Frozen repair thresholds: t_own={fit['t_own']} t_need={fit['t_need']} "
        f"(cost {fit['cost']} when reapplied to the same 50)"
    )
    return config, results


def _strategy_summary(result):
    return {
        "tuned": result["tuned"],
        "comparison_cost": result["comparison_cost"],
        "total_cost": result["total_cost"],
        "frozen_cost": result.get("frozen_cost"),
        "loo_cost": result.get("loo_cost"),
        "n_correct": result["n_correct"],
        "accuracy": result["accuracy"],
        "n_laya_calls": result["n_laya_calls"],
        "n_flips_vs_baseline": result.get("n_flips_vs_baseline"),
        "cost_delta_vs_baseline": result.get("cost_delta_vs_baseline"),
        "settings": result.get("settings"),
    }


def _write_strategy_files(results, results_dir):
    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "strategy", "comparison_cost", "total_cost", "loo_cost", "n_correct",
            "accuracy", "n_laya_calls", "n_flips_vs_baseline", "tuned",
        ])
        writer.writeheader()
        per_path = results_dir / "per_message.csv"
        with per_path.open("w", newline="", encoding="utf-8") as per_handle:
            per_writer = csv.DictWriter(per_handle, fieldnames=[
                "strategy", "id", "label", "prediction", "baseline_prediction",
                "cost", "correct", "flipped", "laya_calls",
            ])
            per_writer.writeheader()
            for result in results:
                writer.writerow({
                    "strategy": result["strategy"],
                    "comparison_cost": result["comparison_cost"],
                    "total_cost": result["total_cost"],
                    "loo_cost": "" if result.get("loo_cost") is None else result["loo_cost"],
                    "n_correct": result["n_correct"],
                    "accuracy": f"{result['accuracy']:.4f}",
                    "n_laya_calls": result["n_laya_calls"],
                    "n_flips_vs_baseline": result.get("n_flips_vs_baseline"),
                    "tuned": result["tuned"],
                })
                for row in result["per_message"]:
                    per_writer.writerow({
                        "strategy": result["strategy"],
                        "id": row["id"],
                        "label": row["label"],
                        "prediction": row["prediction"],
                        "baseline_prediction": row.get("baseline_prediction", ""),
                        "cost": row["cost"],
                        "correct": row["correct"],
                        "flipped": row.get("flipped", ""),
                        "laya_calls": row["laya_calls"],
                    })
                write_json(results_dir / f"{result['strategy']}.json", result)
    print(f"Wrote {summary_path}")


def main():
    args = parse_args()
    messages = load_messages(args.messages) if args.messages else load_development_messages()
    fit_and_write(messages, args.decisions, args.cost_matrix, args.config, args.results)


if __name__ == "__main__":
    main()
