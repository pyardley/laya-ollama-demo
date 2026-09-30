"""Keep the demo route unless both specialists flag an expensive mistake.

Ignore becomes escalate only when ownership says customer and need says staff.
Auto-reply becomes ignore only when ownership says outsider. Escalation is
never overwritten. The confidence gates are fixed at zero: the specialist
choice is enough. Nothing is fitted.

  python experiments/05_costly_miss_repair.py --messages data/customer_messages.json
"""

from cli import run

if __name__ == "__main__":
    run("costly_miss_repair")
