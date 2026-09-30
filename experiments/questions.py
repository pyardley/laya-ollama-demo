"""The Laya questions used in this experiment.

Every question is a 3-option choice. Descriptions are one short sentence so the
rendered option (`label: description`) stays inside Laya's 48-token cap, and
three options stay in the calibrated `choice:3-5` bucket. Nothing here joins
several wordings into one option.

`demo` is the wording already in `laya_demo.py` (a1 / e3 / i9). The other two
questions do a different job from picking the route: who is writing, and what
kind of help they need. Their sentences follow the labelling policy in
`data/customer_messages.json`. They do not quote the nine messages the tuned
wording misses.
"""

DEMO_ID = "demo"
OWNERSHIP_ID = "ownership"
NEED_ID = "need"
REPEAT_ID = "demo_repeat"

# Same instruction and criteria as laya_demo.py.
DEMO = {
    "id": DEMO_ID,
    "type": "choice",
    "instruction": "What action should be taken for this customer support ticket?",
    "criteria": {
        "auto_reply": "A how-to or FAQ: password reset, Bluetooth pairing, order tracking, firmware, or account settings.",
        "escalate_to_human": "Broken hardware, a refund, a safety issue, or an account takeover.",
        "ignore": "Not a customer support request.",
    },
}

# Different job: who is writing. Not a second copy of the three routes.
OWNERSHIP = {
    "id": OWNERSHIP_ID,
    "type": "choice",
    "instruction": "Who is writing, relative to this shop?",
    "criteria": {
        "customer": "A person writing about their own order, account, or device.",
        "outsider": "Spam, a pitch, recruitment, or someone else's notice.",
        "unclear": "Not enough to tell whether the sender is a customer.",
    },
}

# Different job: what kind of help, if any. Not a second copy of the three routes.
NEED = {
    "id": NEED_ID,
    "type": "choice",
    "instruction": "What kind of help does the sender need?",
    "criteria": {
        "steps": "Steps, or the wording of a published policy. Nothing has failed.",
        "staff": "A fault, a refund, a cancellation, a replacement, or an unsafe account.",
        "none": "No help with their own order, account, or device.",
    },
}

QUESTIONS = (DEMO, OWNERSHIP, NEED)

# Collected on the development set so repeated sampling can be checked. Not a
# different question: the same demo call, run again.
REPEAT_OF = DEMO_ID
