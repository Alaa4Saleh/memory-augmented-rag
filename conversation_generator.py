# conversation_generator.py
import json
import random
import os
from typing import List, Dict
import google.generativeai as genai
from config import *
from fact_generator import generate_facts, fact_to_establishment_turn

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Filler question templates
FILLER_QUESTIONS = [
    "What's the weather like today?",
    "Tell me about artificial intelligence",
    "What's in the news?",
    "Explain quantum computing briefly",
    "What do you think about climate change?",
    "How does the internet work?",
    "What's the capital of France?",
    "Tell me an interesting fact about space",
    "What are the benefits of exercise?",
    "How do plants grow?"
]


def generate_memory_questions(facts: List[Dict]) -> Dict[str, List[Dict]]:
    """Generate explicit and implicit questions for each fact using Gemini"""
    questions = {
        "explicit": [],
        "implicit": []
    }

    for fact_obj in facts:
        fact = fact_obj["fact"]
        topic = fact_obj["topic"]

        # Generate explicit question
        explicit_prompt = f"""Given this fact about a user: "{fact}"

Generate ONE direct question that tests if someone remembers this fact.
The question should be simple and straightforward.

Examples:
- Fact: "I'm vegetarian" → Question: "What are my dietary restrictions?"
- Fact: "I work as a teacher" → Question: "What do I do for work?"
- Fact: "I enjoy hiking" → Question: "What outdoor activities do I like?"

Just return the question, nothing else."""

        try:
            explicit_response = model.generate_content(explicit_prompt)
            explicit_q = explicit_response.text.strip().strip('"')
        except Exception as e:
            print(f"Error generating explicit question: {e}")
            explicit_q = f"What did I tell you about {topic}?"

        # Generate implicit question
        implicit_prompt = f"""Given this fact about a user: "{fact}"

Generate ONE indirect question where the user needs to USE this fact to answer correctly.
The question should NOT directly ask about the fact, but the answer requires knowing it.

Examples:
- Fact: "I'm vegetarian" → Question: "Recommend a restaurant for me"
- Fact: "I work remotely" → Question: "Should I move closer to the office?"
- Fact: "I enjoy hiking" → Question: "I'm bored this weekend, any suggestions?"

Just return the question, nothing else."""

        try:
            implicit_response = model.generate_content(implicit_prompt)
            implicit_q = implicit_response.text.strip().strip('"')
        except Exception as e:
            print(f"Error generating implicit question: {e}")
            implicit_q = f"Give me advice related to {topic}"

        questions["explicit"].append({
            "question": explicit_q,
            "related_fact": fact,
            "topic": topic,
            "type": "explicit"
        })

        questions["implicit"].append({
            "question": implicit_q,
            "related_fact": fact,
            "topic": topic,
            "type": "implicit"
        })

    return questions


def generate_gold_answer(question: str, fact: str, question_type: str) -> str:
    """Generate gold standard answer using Gemini"""
    if question_type == "explicit":
        prompt = f"""Answer this question directly based on the given fact:

Fact: {fact}
Question: {question}

Provide a SHORT, direct answer (1-2 sentences max). Be concise."""
    else:  # implicit
        prompt = f"""Answer this question while considering the given fact:

Fact: {fact}
Question: {question}

Provide a helpful answer that respects the fact. Keep it SHORT (2-3 sentences max)."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Based on what you told me ({fact}), here's my answer."


def create_conversation(conv_id: int) -> Dict:
    """Create one complete conversation"""
    print(f"Generating conversation {conv_id}...")

    # 1. Generate facts
    num_facts = random.choice(FACTS_PER_CONVERSATION)
    facts = generate_facts(num_facts)

    # 2. Generate questions
    memory_questions = generate_memory_questions(facts)

    # 3. Build conversation structure
    conversation = {
        "conversation_id": conv_id,
        "facts": facts,
        "turns": []
    }

    turn_num = 1

    # Turns 1-3: Establish facts
    for i, fact_obj in enumerate(facts[:3]):  # Max 3 facts in establishment
        turn = fact_to_establishment_turn(fact_obj, turn_num)
        conversation["turns"].append(turn)
        turn_num += 1

        # Bot acknowledgment
        conversation["turns"].append({
            "turn": turn_num,
            "speaker": "bot",
            "text": "Got it, I'll remember that!",
            "type": "acknowledgment"
        })
        turn_num += 1

    # Remaining turns: filler + probes
    probe_turn_iter = iter(PROBE_TURNS)
    next_probe = next(probe_turn_iter, None)

    while turn_num <= CONVERSATION_LENGTH:
        if turn_num == next_probe:
            # Memory probe turn
            # Alternate between explicit and implicit
            q_type = "explicit" if len([t for t in conversation["turns"] if t.get("is_probe")]) % 2 == 0 else "implicit"
            questions_pool = memory_questions[q_type]

            if questions_pool:
                q_obj = random.choice(questions_pool)
                gold_answer = generate_gold_answer(q_obj["question"], q_obj["related_fact"], q_type)

                conversation["turns"].append({
                    "turn": turn_num,
                    "speaker": "user",
                    "text": q_obj["question"],
                    "type": "memory_probe",
                    "is_probe": True,
                    "probe_type": q_type,
                    "related_fact": q_obj["related_fact"],
                    "topic": q_obj["topic"],
                    "gold_answer": gold_answer
                })

            next_probe = next(probe_turn_iter, None)
        else:
            # Filler turn
            filler_q = random.choice(FILLER_QUESTIONS)
            conversation["turns"].append({
                "turn": turn_num,
                "speaker": "user",
                "text": filler_q,
                "type": "filler",
                "is_probe": False
            })

        turn_num += 1

    return conversation


def generate_dataset():
    """Generate full dataset of conversations"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conversations = []
    for i in range(NUM_CONVERSATIONS):
        conv = create_conversation(i)
        conversations.append(conv)

    # Save to JSON
    with open(CONVERSATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Generated {NUM_CONVERSATIONS} conversations")
    print(f"📁 Saved to: {CONVERSATIONS_FILE}")

    # Print summary
    total_probes = sum(len([t for t in c["turns"] if t.get("is_probe")]) for c in conversations)
    print(f"📊 Total memory probes: {total_probes}")


if __name__ == "__main__":
    generate_dataset()