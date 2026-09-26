# Laya + Ollama: local decision and generation pipeline

A small, fully local demo that pairs two kinds of model:

1. **[Laya](https://huggingface.co/convaiinnovations/laya)** - an open-source "System 1" decision model (an alternative to TypeSafe AI's Jev). It scores a customer-support ticket against three predefined actions and returns a choice with a probability for each one.
2. **[Ollama](https://ollama.com/)** running `llama3.1:8b` - only called when the chosen action needs a written reply.

Everything runs on CPU. No API keys are needed, and no ticket data leaves the machine.

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
python laya_demo.py
python laya_demo.py "How do I reset my password?"   # try your own ticket
```

The Laya weights (`convaiinnovations/laya`) download from Hugging Face on first run.

## Routing

Each Laya decision leads to a different action:

| Condition | What happens |
| --- | --- |
| `answer_confidence` below 70% (`CONFIDENCE_THRESHOLD`) | Sent to a human for review, whatever Laya chose. Ollama drafts an **internal handoff note** that includes Laya's probability spread. **No customer reply is sent.** |
| `auto_reply` | Ollama drafts a **customer reply ready to send**: body only, no placeholders. |
| `escalate_to_human` | The customer gets a **fixed holding reply** (no LLM involved), and Ollama drafts an **internal handoff note** for the agent. |
| `ignore` | Nothing. |

## Output

```text
============================================================
 🤖 LAYA DECISION ANALYSIS
============================================================
Primary Recommendation : ESCALATE_TO_HUMAN
Confidence Score       : 78.1%

Probability Breakdown:
  • escalate_to_human  [███████████████████████░░░░░░░]  78.1%  👈 (Selected)
  • auto_reply         [███░░░░░░░░░░░░░░░░░░░░░░░░░░░]  12.0%
  • ignore             [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   9.8%

============================================================
 🚀 EXECUTION ROUTE
============================================================
Action triggered: Escalating to a human agent.

------------------------------------------------------------
 HOLDING REPLY TO CUSTOMER
------------------------------------------------------------
...
------------------------------------------------------------
 OLLAMA HANDOFF NOTE (internal, for the agent)
------------------------------------------------------------
**Summary:** Customer reports speaker crackling when playing audio.
...
```

## Which confidence to use

A Laya `choice` answer has two confidence fields:

| Field | What it is | Calibrated? |
| --- | --- | --- |
| `answer_confidence` | `max(p)` - the probability of the chosen answer | Yes - safe to threshold on |
| `confidence` | Normalised entropy, `1 - H(p) / ln(k)` - how concentrated the whole distribution is | No |

For the ticket above, `confidence` is 38.5% and `answer_confidence` is 78.1%. This script uses `answer_confidence`.

## Known limitations

- **Laya chose `escalate_to_human` for every ticket tried so far.** That includes a password-reset FAQ (61.5%) and obvious spam (43.8%, only just ahead of `ignore` at 42.2%). The confidence threshold catches both, but the `auto_reply` route has not yet been triggered by a real ticket. The option descriptions in `criteria` probably need tuning.
- **Laya prints a `RuntimeWarning` on load.** In v0.3.20 it says the checkpoint's temperature for `choice` questions with 11 or more options is out of range, and treats that bucket as uncalibrated. This demo's three-option question isn't affected.
- **Handoff notes are LLM output and can misread the ticket.** For the spam example, the note treated the sender as a customer who had *received* a phishing email.

## Write-up

Full walkthrough: [A Local Decision + Generation Pipeline with Laya and Ollama](https://paulyardleyqa.co.uk/blog/laya-ollama-local-pipeline/)
