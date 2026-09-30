"""Two specialist questions, each a different job from choosing the route.

`ownership` asks who is writing. `need` asks what kind of help they need.
A fixed map turns those answers into a route. The demo wording is used only
when that map does not name one. No threshold is fitted.

This is not a second copy of the three route labels. Re-asking auto_reply /
escalate / ignore in different words was already tried, and it did not move
the nine misses.

  python experiments/03_specialist_route.py --messages data/customer_messages.json
"""

from cli import run

if __name__ == "__main__":
    run("specialist_route")
