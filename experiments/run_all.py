"""Collect Laya decisions on the development set, fit the frozen config, and score every rule.

  python experiments/run_all.py
  python experiments/run_all.py --messages data/customer_messages.json

A new labelled file is not passed here. Score that with experiments/apply.py
so the thresholds fitted on these 50 messages stay frozen.
"""

import argparse
from pathlib import Path

from common import DEFAULT_MESSAGES, is_development_set, load_messages, print_result
from collect import collect_decisions
from fit_config import fit_and_write


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    messages = load_messages(args.messages)
    if not is_development_set(messages):
        raise SystemExit(
            "run_all.py fits thresholds, so it only accepts the original 50 messages. "
            "For a new file run: python experiments/apply.py --messages <file> --config experiments/config.json --all"
        )
    collect_decisions(
        messages,
        Path("experiments/results/raw_decisions.json"),
        batch_size=args.batch_size,
        device=args.device,
        fresh=args.fresh,
    )
    _config, results = fit_and_write(
        messages,
        Path("experiments/results/raw_decisions.json"),
        Path("data/cost_matrix.csv"),
        Path("experiments/config.json"),
        Path("experiments/results"),
    )
    print()
    for result in results:
        print_result(result)
        print()


if __name__ == "__main__":
    main()
