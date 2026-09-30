"""Two calls of the same demo question, then a majority vote.

Laya's forward pass on CPU is expected to be deterministic, so the second call
should match the first and the vote should tie the baseline at twice the cost
in calls. The script is kept so that check is reproducible. It is not a search
over temperature or dropout.

  python experiments/02_repeated_majority.py --messages data/customer_messages.json
"""

from cli import run

if __name__ == "__main__":
    run("repeated_majority")
