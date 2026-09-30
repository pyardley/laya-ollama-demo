"""Single-decision baseline: one Laya call, wording a1 / e3 / i9, highest probability.

This is the rule in laya_demo.py. The published development cost is 19. The
same call is also scored with two controls that the criteria write-up already
rejected: a 70% confidence gate, and lowest expected cost. Those controls are
printed inside the result and are not candidates.

  python experiments/01_baseline.py
  python experiments/01_baseline.py --messages path/to/new_messages.csv --decisions path/to/raw_decisions.json --config experiments/config.json
"""

from cli import run

if __name__ == "__main__":
    run("baseline_argmax")
