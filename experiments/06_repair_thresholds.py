"""The repair rule with confidence thresholds frozen from the development set.

fit_config.py chose the thresholds by minimising cost on the original 50.
This script reads them from experiments/config.json and does not refit.
On the development set it also reports leave-one-out cost, where each
message is scored with thresholds fit on the other 49 only. On any other
file the leave-one-out refit is skipped.

  python experiments/06_repair_thresholds.py --messages data/customer_messages.json
  python experiments/06_repair_thresholds.py --messages path/to/new_messages.csv --decisions path/to/raw_decisions.json
"""

from cli import run

if __name__ == "__main__":
    run("repair_thresholds")
