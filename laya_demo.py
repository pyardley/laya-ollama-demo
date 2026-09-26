import time

# Start the clock before the heavy imports: importing torch/transformers is a real cost.
_script_start = time.perf_counter()
timings = {}  # phase name -> seconds, in the order the phases ran


class timed:
    """Context manager that records how long a block took under `timings[name]`."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *exc):
        timings[self.name] = timings.get(self.name, 0) + time.perf_counter() - self.start


import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

with timed("Import libraries (torch, transformers, langchain)"):
    from laya import load
    from langchain_ollama import OllamaLLM

# Decisions below this calibrated confidence are sent to a human, whatever Laya chose.
CONFIDENCE_THRESHOLD = 0.70

HOLDING_REPLY = """Hello,

Thank you for getting in touch. Your message has been passed to one of our support
specialists, who will contact you directly to help resolve this.

Customer Support Team"""

# 1. Load Models
print("Loading models...")
with timed("Load Laya model (download check + weights)"):
    agent = load("convaiinnovations/laya", device="cpu")
# Only creates a client - Ollama loads llama3.1 into memory on the first call.
ollama_llm = OllamaLLM(model="llama3.1:8b")

# Pass a ticket on the command line to try other routes, e.g.
#   python laya_demo.py "How do I reset my password?"
ticket = sys.argv[1] if len(sys.argv) > 1 else "My speaker crackles when playing audio."

# 2. Perform Decision Pass
with timed("Laya decision (predict)"):
    decision = agent.predict(
        state=ticket,
        questions={
            "action": {
                "instructions": "What action should be taken for this customer support ticket?",
                "type": "choice",
                "criteria": {
                    "auto_reply": "The issue is simple and can be addressed automatically.",
                    "escalate_to_human": "The hardware issue or replacement request needs a human agent.",
                    "ignore": "The message is spam or completely irrelevant."
                }
            }
        }
    )

# 3. Extract Results
action_data = decision["answers"]["action"]
chosen_action = action_data["choice"]
probabilities = action_data["probabilities"]
# 'answer_confidence' is max(p), the calibrated value that is safe to threshold on.
# 'confidence' is normalised entropy over all options and is NOT calibrated.
answer_confidence = action_data.get("answer_confidence", 0)

# 4. Formatted Display
print("\n" + "=" * 60)
print(" 📥 INPUT TICKET")
print("=" * 60)
print(f'"{ticket}"\n')

print("=" * 60)
print(" 🤖 LAYA DECISION ANALYSIS")
print("=" * 60)
print(f"Primary Recommendation : {chosen_action.upper()}")
print(f"Confidence Score       : {answer_confidence * 100:.1f}%\n")

print("Probability Breakdown:")
for option, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
    bar_length = int(prob * 30)
    bar = "█" * bar_length + "░" * (30 - bar_length)
    indicator = "👈 (Selected)" if option == chosen_action else ""
    print(f"  • {option:<18} [{bar}] {prob * 100:5.1f}%  {indicator}")

print("\n" + "=" * 60)
print(" 🚀 EXECUTION ROUTE")
print("=" * 60)


def print_block(title, text):
    print("-" * 60)
    print(f" {title}")
    print("-" * 60)
    print(text.strip())
    print("-" * 60)


def ask_ollama(prompt):
    """invoke() equivalent that also records Ollama's own timing breakdown.

    Ollama reports durations in nanoseconds. The first call of a run includes loading
    llama3.1 into memory ("load"), which can dwarf the actual generation time.
    """
    with timed("Ollama call (wall clock)"):
        result = ollama_llm.generate([prompt])
    generation = result.generations[0][0]
    info = generation.generation_info or {}
    for key, label in [("load_duration", "  of which: Ollama model load"),
                       ("prompt_eval_duration", "  of which: Ollama prompt processing"),
                       ("eval_duration", "  of which: Ollama text generation")]:
        if info.get(key):
            timings[label] = timings.get(label, 0) + info[key] / 1e9
    if info.get("eval_count") and info.get("eval_duration"):
        print(f"[Ollama generated {info['eval_count']} tokens at "
              f"{info['eval_count'] / (info['eval_duration'] / 1e9):.1f} tokens/s]\n")
    return generation.text


def draft_handoff_note(context):
    prompt = (
        f"You are helping triage a customer support ticket for a human agent. The ticket says: '{ticket}'.\n"
        f"{context}\n"
        "Write a concise internal handoff note for the agent - not a reply to the customer. "
        "Use exactly these headings:\n"
        "Summary: (one line)\n"
        "Likely cause:\n"
        "Questions to ask the customer:\n"
        "Suggested next step:\n"
        "Output only the note, with no preamble."
    )
    print_block("OLLAMA HANDOFF NOTE (internal, for the agent)", ask_ollama(prompt))


# 5. Route Execution
if answer_confidence < CONFIDENCE_THRESHOLD:
    # Not sure enough to act on any route - including telling the customer anything.
    # A human decides; they get Laya's full probability spread as context.
    print(f"Confidence {answer_confidence:.1%} is below the {CONFIDENCE_THRESHOLD:.0%} threshold: "
          "sending to a human for review. No customer reply sent.\n")
    spread = ", ".join(f"{k} {v:.0%}" for k, v in probabilities.items())
    print("Invoking Ollama to draft an internal handoff note...\n")
    draft_handoff_note(f"An automated classifier was unsure how to handle it ({spread}).")

elif chosen_action == "auto_reply":
    print("Action triggered: Invoking Ollama to draft an automatic customer reply...\n")

    prompt = (
        f"A customer reported: '{ticket}'. Write a short, polite, professional support email "
        "reply that resolves the issue. Output only the email body, with no preamble. "
        "Do not use placeholders such as [Customer's Name]; open with 'Hello,' and sign off "
        "as 'Customer Support Team'."
    )
    print_block("OLLAMA AUTO-REPLY (ready to send)", ask_ollama(prompt))

elif chosen_action == "escalate_to_human":
    print("Action triggered: Escalating to a human agent.\n")

    # A fixed holding reply: nothing for the LLM to get wrong, and no promises about a fix.
    print_block("HOLDING REPLY TO CUSTOMER", HOLDING_REPLY)

    print("\nInvoking Ollama to draft an internal handoff note...\n")
    draft_handoff_note("It has been escalated to you by an automated classifier.")

else:
    print("Action evaluated as 'ignore'. No response generated.")

# 6. Timing Summary
total = time.perf_counter() - _script_start
print("\n" + "=" * 60)
print(" ⏱️ TIMING")
print("=" * 60)
for name, seconds in timings.items():
    print(f"  {name:<50} {seconds:7.2f}s")
accounted = sum(s for n, s in timings.items() if not n.startswith("  of which"))
print(f"  {'Everything else (printing, small imports)':<50} {total - accounted:7.2f}s")
print(f"  {'TOTAL':<50} {total:7.2f}s")

print("\n" + "=" * 60 + "\n")