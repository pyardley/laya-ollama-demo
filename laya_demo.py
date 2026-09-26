import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
agent = load("convaiinnovations/laya", device="cpu")
ollama_llm = OllamaLLM(model="llama3.1:8b")

# Pass a ticket on the command line to try other routes, e.g.
#   python laya_demo.py "How do I reset my password?"
ticket = sys.argv[1] if len(sys.argv) > 1 else "My speaker crackles when playing audio."

# 2. Perform Decision Pass
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
    print_block("OLLAMA HANDOFF NOTE (internal, for the agent)", ollama_llm.invoke(prompt))


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
    print_block("OLLAMA AUTO-REPLY (ready to send)", ollama_llm.invoke(prompt))

elif chosen_action == "escalate_to_human":
    print("Action triggered: Escalating to a human agent.\n")

    # A fixed holding reply: nothing for the LLM to get wrong, and no promises about a fix.
    print_block("HOLDING REPLY TO CUSTOMER", HOLDING_REPLY)

    print("\nInvoking Ollama to draft an internal handoff note...\n")
    draft_handoff_note("It has been escalated to you by an automated classifier.")

else:
    print("Action evaluated as 'ignore'. No response generated.")

print("\n" + "=" * 60 + "\n")