"""Majority of the demo vote and the two specialist votes.

The demo choice is kept when the vote ties and the demo is part of the tie,
so the specialists change the route only when they outvote it together.
No weight is fitted. Averaging probabilities of nearby route-wordings was
already tried and repeated the same nine misses; this vote uses different
questions, not those wordings.

  python experiments/04_mapped_majority.py --messages data/customer_messages.json
"""

from cli import run

if __name__ == "__main__":
    run("mapped_majority")
