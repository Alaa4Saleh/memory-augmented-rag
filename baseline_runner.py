# baseline_runner.py
import json
import google.generativeai as genai
from config import *

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


def baseline_no_memory(conversation: dict, turn: dict) -> str:
    """Baseline: Answer with NO conversation history"""
    question = turn["text"]

    prompt = f"""Answer this question briefly (1-2 sentences):

Question: {question}

If you don't have information, say "I don't have that information." """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"


def baseline_last_k(conversation: dict, turn: dict, k: int = 5) -> str:
    """Baseline: Use last K turns as context"""
    question = turn["text"]
    turn_num = turn["turn"]

    # Get last k turns before current turn
    history_turns = [t for t in conversation["turns"] if t["turn"] < turn_num]
    last_k_turns = history_turns[-k:] if len(history_turns) > k else history_turns

    # Build context
    context = "\n".join([f"{t['speaker']}: {t['text']}" for t in last_k_turns])

    prompt = f"""Based on this conversation history, answer the question:

Conversation:
{context}

Question: {question}

Answer briefly (1-2 sentences). If the history doesn't contain the information, say so."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"


def run_baseline(system_name: str, system_func):
    """Run a baseline system on all conversations"""
    print(f"\n{'=' * 50}")
    print(f"Running: {system_name}")
    print(f"{'=' * 50}")

    # Load conversations
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    predictions = []

    for conv in conversations:
        print(f"\nConversation {conv['conversation_id']}...")

        for turn in conv["turns"]:
            # Only generate predictions for memory probes
            if not turn.get("is_probe"):
                continue

            print(f"  Turn {turn['turn']}: {turn['text'][:50]}...")

            # Generate prediction
            if system_name == "no_memory":
                prediction = baseline_no_memory(conv, turn)
            elif system_name == "last_5":
                prediction = system_func(conv, turn, k=5)
            else:
                prediction = system_func(conv, turn)

            predictions.append({
                "conversation_id": conv["conversation_id"],
                "turn": turn["turn"],
                "question": turn["text"],
                "probe_type": turn["probe_type"],
                "related_fact": turn["related_fact"],
                "gold_answer": turn["gold_answer"],
                "prediction": prediction,
                "system": system_name
            })

    return predictions


def main():
    """Run all baseline systems"""
    all_predictions = []

    # System 1: No memory
    preds1 = run_baseline("no_memory", baseline_no_memory)
    all_predictions.extend(preds1)

    # System 2: Last 5 turns
    preds2 = run_baseline("last_5", baseline_last_k)
    all_predictions.extend(preds2)

    # Save predictions
    with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Predictions saved to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()