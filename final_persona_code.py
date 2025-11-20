import json
import random
import time
from pathlib import Path
from dotenv import load_dotenv
import os

# -------------------------------------
# CONFIG
# -------------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")      # from .env file
INPUT_PERSONAS = Path("data/persona_chat_clean.json")
OUTPUT_PATH = Path("data/memory_eval_dataset.json")

NUM_CONVERSATIONS = 30
MIN_TURNS = 20
MAX_TURNS = 30

random.seed(42)


# -------------------------------------
# LLM CALL (NO MODEL NAME)
# -------------------------------------
def llm(prompt, max_tokens=350):
    """
    Calls any strong API using only the API key.
    No model name is exposed here by design.
    You can plug in any client library.
    """

    import openai
    openai.api_key = API_KEY

    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message["content"].strip()

        except Exception:
            time.sleep(1)

    return "ERROR"


# -------------------------------------
# FILLER MESSAGE GENERATOR
# -------------------------------------
def gen_filler_turn(persona_facts, history):
    """
    Generates a natural, persona-consistent user message.
    High-quality, clean English. No inappropriate content.
    """
    prompt = f"""
You are generating a single USER message for a conversation.

Persona facts (this is the user's self-description):
{persona_facts}

Conversation so far:
{history}

Generate ONE new user message that:
- Is natural, coherent, everyday conversation
- Aligns with the persona facts (indirectly)
- No sexual, violent, political, hateful content
- Not repetitive or generic like "okay" or "thanks"
- Should NOT explicitly restate the persona facts
- Should sound like a real human continuing the conversation
- 1–2 sentences max
Only output the message, nothing else.
"""
    return llm(prompt)


# -------------------------------------
# PROBE GENERATION (EXPLICIT + IMPLICIT)
# -------------------------------------
def gen_probe(persona_facts, history):
    """
    Generates either:
    - an explicit memory probe (direct recall)
    - an implicit memory probe (indirect recall)

    EXACT rules based on your final instructions:
      - Explicit example: "What diet did I say I am on?"
      - Implicit example: "Suggest a meal recipe for me"
    """
    prompt = f"""
Create ONE memory probe question for a memory-evaluation dataset.

Persona facts:
{persona_facts}

Conversation so far:
{history}

Rules:
1. The probe must test memory of a FACT from the persona.
2. Two types:
   - EXPLICIT (direct recall):
       e.g., "What diet did I say I'm on?" for a vegetarian fact
   - IMPLICIT (indirect recall):
       e.g., "Can you suggest a meal recipe for me?" where the correct answer must respect that the user is vegetarian
3. The probe must be phrased naturally and must refer to *something the user said earlier*.
4. Output ONLY the probe question, nothing else.
"""
    return llm(prompt)


# -------------------------------------
# GOLD ANSWER + KEYWORDS
# -------------------------------------
def gen_gold_answer(persona_facts, probe):
    """
    Gold answers must:
      - Correctly reflect the persona fact being tested
      - Remain concise and unambiguous
      - Provide a few strong keywords for later scoring
    """
    prompt = f"""
Persona facts:
{persona_facts}

Probe question:
{probe}

TASK:
1. Write the correct GOLD ANSWER strictly based on the persona facts.
2. Then extract 2–5 KEYWORDS that represent the essential content of that answer.
   - If the persona is vegetarian and the probe is implicit (recipe suggestion),
     the gold answer should implicitly reflect a vegetarian-safe recipe.
3. Follow this EXACT format:

ANSWER: <answer text>
KEYWORDS: ["keyword1", "keyword2", ...]

Only output this exact format.
"""
    return llm(prompt)


# -------------------------------------
# LOAD PERSONAS FROM FILE
# -------------------------------------
def load_personas():
    with INPUT_PERSONAS.open("r", encoding="utf-8") as f:
        data = json.load(f)

    personas = []
    for p in data:
        if "persona" in p:
            cleaned = [x for x in p["persona"] if isinstance(x, str)]
            text = " ".join(cleaned).strip()
            if text:
                personas.append(text)

    return personas


# -------------------------------------
# CONVERSATION GENERATOR
# -------------------------------------
def build_conversation(conv_id, persona):
    history = []
    turns = []

    turn_count = random.randint(MIN_TURNS, MAX_TURNS)

    for t in range(1, turn_count + 1):
        # Decide if this is a probe turn
        if t % 5 == 0:  # Exactly as you required
            probe = gen_probe(persona, history)
            gold = gen_gold_answer(persona, probe)

            # Parse gold answer
            lines = gold.split("\n")
            ans = lines[0].replace("ANSWER:", "").strip()
            kws = lines[1].replace("KEYWORDS:", "").strip()

            turns.append({
                "turn": t,
                "type": "probe",
                "prompt": probe,
                "gold_answer": ans,
                "keywords": json.loads(kws)
            })

            history.append(probe)

        else:
            filler = gen_filler_turn(persona, history)
            turns.append({
                "turn": t,
                "type": "filler",
                "prompt": filler
            })
            history.append(filler)

    return {
        "conversation_id": f"eval_{conv_id:04d}",
        "persona": persona,
        "turns": turns
    }


# -------------------------------------
# MAIN
# -------------------------------------
def main():
    personas = load_personas()

    dataset = []
    for i in range(NUM_CONVERSATIONS):
        persona = random.choice(personas)
        conv = build_conversation(i + 1, persona)
        dataset.append(conv)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
