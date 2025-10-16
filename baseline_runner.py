# baseline_runner.py
import json
import time
import google.generativeai as genai
from config import *

genai.configure(api_key=GOOGLE_API_KEY)

# Rate limiting
REQUESTS_PER_MINUTE = 9
request_timestamps = []


def wait_if_needed():
    """Rate limiting"""
    global request_timestamps
    current_time = time.time()
    request_timestamps = [t for t in request_timestamps if current_time - t < 60]

    if len(request_timestamps) >= REQUESTS_PER_MINUTE:
        sleep_time = 60 - (current_time - request_timestamps[0]) + 1
        if sleep_time > 0:
            print(f"  ⏳ Waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            request_timestamps.clear()

    request_timestamps.append(time.time())


def call_model_with_retry(chat_session, message, max_retries=3):
    """Send message in chat with retry"""
    for attempt in range(max_retries):
        wait_if_needed()
        try:
            response = chat_session.send_message(message)
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"  ⚠️ Quota hit, waiting 60s...")
                time.sleep(60)
                request_timestamps.clear()
            else:
                return f"Error: {error_msg[:100]}"
    return "Error: Max retries exceeded"


def run_normal_chat_baseline():
    """
    Chat with the model normally, turn by turn.
    The model maintains conversation context naturally (like ChatGPT).
    """
    print("\n" + "=" * 70)
    print("🤖 BASELINE: Normal Chat (Default Model Memory)")
    print("=" * 70)
    print("Each conversation is a separate chat session.")
    print("Model remembers within each session naturally.")
    print("=" * 70)

    # Load conversations
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    all_predictions = []

    for conv in conversations:
        print(f"\n{'=' * 70}")
        print(f"💬 Starting Conversation {conv['conversation_id']}")
        print(f"{'=' * 70}")
        print(f"Facts to be mentioned:")
        for fact in conv["facts"]:
            print(f"  - [{fact['topic']}] {fact['fact']}")

        # Start a NEW chat session for this conversation
        model = genai.GenerativeModel(MODEL_NAME)
        chat = model.start_chat(history=[])

        print(f"\n🔄 Processing {len(conv['turns'])} turns...")

        for turn in conv["turns"]:
            turn_num = turn["turn"]
            speaker = turn["speaker"]
            text = turn["text"]

            if speaker == "user":
                print(f"\n  Turn {turn_num}")
                print(f"  👤 User: {text[:70]}...")

                # Send message to model
                response = call_model_with_retry(chat, text)

                print(f"  🤖 Assistant: {response[:70]}...")

                # If this is a memory probe, save for evaluation
                if turn.get("is_probe"):
                    print(f"  📌 [MEMORY PROBE - {turn['probe_type']}]")
                    print(f"     Testing recall of: {turn['related_fact']}")

                    all_predictions.append({
                        "conversation_id": conv["conversation_id"],
                        "turn": turn_num,
                        "question": text,
                        "probe_type": turn["probe_type"],
                        "related_fact": turn["related_fact"],
                        "topic": turn["topic"],
                        "gold_answer": turn["gold_answer"],
                        "prediction": response,
                        "system": "normal_chat"
                    })

            elif speaker == "bot":
                # Bot acknowledgment turn - send it to maintain flow
                print(f"\n  Turn {turn_num}")
                print(f"  👤 User: (continues conversation)")

                # We don't actually need to send bot turns since they're not questions
                # But if you want to include them:
                # response = call_model_with_retry(chat, text)
                pass

        print(f"\n  ✅ Conversation {conv['conversation_id']} complete")

    # Save predictions
    with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"✅ ALL CONVERSATIONS COMPLETE")
    print(f"📊 Total memory probes: {len(all_predictions)}")
    print(f"💾 Saved to: {PREDICTIONS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_normal_chat_baseline()