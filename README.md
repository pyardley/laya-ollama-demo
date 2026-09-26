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
```

The Laya weights (`convaiinnovations/laya`) download from Hugging Face on first run.

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
```

## Which confidence to use

A Laya `choice` answer has two confidence fields:

| Field | What it is | Calibrated? |
| --- | --- | --- |
| `answer_confidence` | `max(p)` - the probability of the chosen answer | Yes - safe to threshold on |
| `confidence` | Normalised entropy, `1 - H(p) / ln(k)` - how concentrated the whole distribution is | No |

For the ticket above, `confidence` is 38.5% and `answer_confidence` is 78.1%. This script uses `answer_confidence`.

## Known limitations

- **Laya prints a `RuntimeWarning` on load.** In v0.3.20 it says the checkpoint's temperature for `choice` questions with 11 or more options is out of range, and treats that bucket as uncalibrated. This demo's three-option question isn't affected.
- **`auto_reply` and `escalate_to_human` take the same path.** Both draft a customer email.
- **The LLM draft includes placeholders** like `[Customer's Name]` and needs review before sending.

## Write-up

Full walkthrough: [A Local Decision + Generation Pipeline with Laya and Ollama](https://paulyardleyqa.co.uk/blog/laya-ollama-local-pipeline/)
