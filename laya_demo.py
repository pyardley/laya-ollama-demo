import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from laya import load
from langchain_ollama import OllamaLLM

# 1. Load Models
print("Loading models...")
agent = load("convaiinnovations/laya", device="cpu")
ollama_llm = OllamaLLM(model="llama3.1:8b")

ticket = "My speaker crackles when playing audio."

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

# 4. Formatted Display
print("\n" + "=" * 60)
print(" 📥 INPUT TICKET")
print("=" * 60)
print(f'"{ticket}"\n')

print("=" * 60)
print(" 🤖 LAYA DECISION ANALYSIS")
print("=" * 60)
print(f"Primary Recommendation : {chosen_action.upper()}")
# 'answer_confidence' is max(p), the calibrated value that is safe to threshold on.
# 'confidence' is normalised entropy over all options and is NOT calibrated.
print(f"Confidence Score       : {action_data.get('answer_confidence', 0) * 100:.1f}%\n")

print("Probability Breakdown:")
for option, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
    bar_length = int(prob * 30)
    bar = "█" * bar_length + "░" * (30 - bar_length)
    indicator = "👈 (Selected)" if option == chosen_action else ""
    print(f"  • {option:<18} [{bar}] {prob * 100:5.1f}%  {indicator}")

print("\n" + "=" * 60)
print(" 🚀 EXECUTION ROUTE")
print("=" * 60)

# 5. Route Execution
if chosen_action in ["auto_reply", "escalate_to_human"]:
    print(f"Action triggered: Invoking Ollama to draft response...\n")

    prompt = f"A customer reported: '{ticket}'. Write a short, polite, professional support email response addressing this issue."
    response = ollama_llm.invoke(prompt)

    print("-" * 60)
    print(" OLLAMA GENERATED DRAFT")
    print("-" * 60)
    print(response.strip())
    print("-" * 60)
else:
    print("Action evaluated as 'ignore'. No auto-response generated.")

print("\n" + "=" * 60 + "\n")