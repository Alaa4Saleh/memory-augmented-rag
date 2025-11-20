"""
============================================================
  PERSONA MEMORY DATASET — LONG CONVERSATIONS VERSION
============================================================

- Builds long (100–120 turn) conversations per persona.
- Based on Persona-Chat (AlekseyKorshuk/persona-chat).
- Uses several persona facts and revisits them multiple times
  with varied explicit/implicit memory probes.
- Uses Gemini to create gold answers + keywords.
- Keeps only English, meaningful, non-toxic turns.
- Output format is:

{
  "conversation_id": "persona_long_0001",
  "persona": [...],
  "turns": [
      { "turn": 1, "type": "filler", "prompt": "..." },
      ...
      {
        "turn": 5,
        "type": "explicit_probe",
        "prompt": "...",
        "gold_answer": "...",
        "keywords": ["..."]
      }
  ]
}
============================================================
"""

import os
import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from langdetect import detect
import google.generativeai as genai
from tqdm import tqdm
from datasets import load_dataset


# ============================================================
# 1) CONFIG + API
# ============================================================

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

MAX_CONVS = 50

MIN_TURNS_PER_CONV = 100
MAX_TURNS_PER_CONV = 120

PROBE_EVERY = 6

MAX_TOTAL_PROBES_PER_CONV = 25
MAX_PROBES_PER_FACT = 4
BAD_WORDS = {
    "fuck", "shit", "bitch", "asshole", "slut", "dick",
    "cock", "pussy", "rape", "porn", "nigger", "faggot",
}


# ============================================================
# 2) LOAD PERSONA-CHAT
# ============================================================

print("📥 Loading PersonaChat...")
ds = load_dataset("AlekseyKorshuk/persona-chat", split="train")
print(f"Loaded {len(ds)} persona entries.")


# ============================================================
# 3) QUALITY HELPERS
# ============================================================

def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return False


def contains_bad_language(text: str) -> bool:
    low = text.lower()
    return any(bw in low for bw in BAD_WORDS)


def is_meaningful(text: str) -> bool:
    text = text.strip()
    if len(text) < 6:
        return False
    if text.lower() in {"ok", "lol", "lmao", "idk", "??", "?", "k"}:
        return False
    return True


# ============================================================
# 4) GEMINI HELPER
# ============================================================

def safe_gemini(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 200,
                },
            )
            if getattr(r, "text", None):
                return r.text.strip()
        except Exception as e:
            print(f"⚠ Gemini error: {e} (attempt {attempt+1})")
            # backoff קטן
            time.sleep(2 + attempt)
    return ""


# ============================================================
# 5) SMART QUESTION TEMPLATES
# ============================================================

EXPLICIT_TEMPLATES = [
    "Earlier you learned something about me: '{}'. How would you describe that in your own words?",
    "Thinking back on what I shared about myself — specifically '{}', what was it about?",
    "Can you recall the detail I told you earlier about '{}'?",
    "What was the personal detail I previously mentioned about '{}'?",
    "From my earlier introduction, what did I say about '{}'?",
    "If you remember my persona, how would you restate the fact: '{}'?"
]

IMPLICIT_TEMPLATES = [
    "Given that earlier I mentioned '{}', how might that influence my behavior in situations like this?",
    "If you consider what I told you before about '{}', what would you infer about my preferences?",
    "Based on my earlier comment about '{}', what conclusion might you draw?",
    "Considering the detail I shared about '{}', how do you think it reflects on my personality?",
    "Knowing that '{}', what kind of choices do you think I am more likely to make?",
    "If you keep in mind '{}', how might that shape my reactions to different events?"
]


def smart_explicit_question(fact: str) -> str:
    """Ask about a fact directly but naturally, using varied templates."""
    template = random.choice(EXPLICIT_TEMPLATES)
    return template.format(fact)


def smart_implicit_question(fact: str) -> str:
    """Ask indirectly about a persona fact."""
    template = random.choice(IMPLICIT_TEMPLATES)
    return template.format(fact)


def gold_answer_prompt(fact: str, question: str) -> str:
    """Tell Gemini to produce a gold answer + keywords."""
    return (
        "A user has the following persona fact:\n"
        f"'{fact}'\n\n"
        f"They now ask: {question}\n\n"
        "Provide:\n"
        "Answer: <one sentence summarizing the correct factual recall or inference>\n"
        "Keywords: <3-6 important words or phrases, comma separated>"
    )


# ============================================================
# 6) CONVERSATION BUILDING (LONG VERSION)
# ============================================================

def extract_filler_candidates(utterances):
    """
    Extract user messages from PersonaChat utterances and filter them.
    Returns a shuffled list of filler texts.
    """
    filler = []
    for utt in utterances:
        items = utt["history"] + [utt.get("text", "")]
        for line in items:
            if "user:" in line:
                text = line.split(":", 1)[1].strip()
                if (
                    is_meaningful(text)
                    and is_english(text)
                    and not contains_bad_language(text)
                ):
                    filler.append(text)
    random.shuffle(filler)
    return filler


def choose_focus_facts(persona):
    """
    Choose several persona facts to repeatedly probe.
    """
    facts = [
        f.strip()
        for f in persona
        if isinstance(f, str) and f.strip()
    ]
    if not facts:
        return []

    num_focus = min(4, len(facts))
    return random.sample(facts, num_focus)


def build_persona_conversation_long(persona, utterances):
    """
    Build a single long conversation (100–120 turns) for one persona.
    Includes many filler turns and multiple probes per fact.
    """
    filler = extract_filler_candidates(utterances)

    if len(filler) < 70:
        return None

    focus_facts = choose_focus_facts(persona)
    if not focus_facts:
        return None

    fact_probe_counts = {f: 0 for f in focus_facts}
    total_probes = 0

    target_turns = random.randint(MIN_TURNS_PER_CONV, MAX_TURNS_PER_CONV)

    turns = []
    turn_number = 1
    ptr = 0

    max_probes_this_conv = min(
        MAX_TOTAL_PROBES_PER_CONV,
        target_turns // (PROBE_EVERY + 1)
    )

    while turn_number <= target_turns and ptr < len(filler):

        turns.append({
            "turn": turn_number,
            "type": "filler",
            "prompt": filler[ptr]
        })
        ptr += 1
        turn_number += 1

        if turn_number > target_turns:
            break

        if (
            len(turns) >= PROBE_EVERY
            and len(turns) % PROBE_EVERY == 0
            and total_probes < max_probes_this_conv
        ):
            candidates = [
                f for f in focus_facts
                if fact_probe_counts[f] < MAX_PROBES_PER_FACT
            ]
            if not candidates:
                continue

            fact = min(
                candidates,
                key=lambda f: fact_probe_counts[f]
            )

            probe_type = random.choice(["explicit_probe", "implicit_probe"])
            if probe_type == "explicit_probe":
                question = smart_explicit_question(fact)
            else:
                question = smart_implicit_question(fact)

            g_prompt = gold_answer_prompt(fact, question)
            g_out = safe_gemini(g_prompt)

            gold_answer = ""
            keywords = []

            for line in g_out.splitlines():
                low = line.lower().strip()
                if low.startswith("answer:"):
                    gold_answer = line.split(":", 1)[1].strip()
                elif low.startswith("keywords:"):
                    keywords = [
                        k.strip()
                        for k in line.split(":", 1)[1].split(",")
                        if k.strip()
                    ]

            turns.append({
                "turn": turn_number,
                "type": probe_type,
                "prompt": question,
                "gold_answer": gold_answer,
                "keywords": keywords,
                "persona_fact": fact
            })
            turn_number += 1
            total_probes += 1
            fact_probe_counts[fact] += 1

    if len(turns) < MIN_TURNS_PER_CONV * 0.8:
        return None

    return turns


# ============================================================
# 7) MAIN LOOP
# ============================================================

final_data = []

for idx, row in enumerate(tqdm(ds, desc="Building LONG persona conversations")):
    persona = row["personality"]
    utterances = row["utterances"]

    conv_turns = build_persona_conversation_long(persona, utterances)
    if conv_turns is None:
        continue

    final_data.append({
        "conversation_id": f"persona_long_{idx:04d}",
        "persona": persona,
        "turns": conv_turns
    })

    if len(final_data) >= MAX_CONVS:
        break

print(f"🧠 Built {len(final_data)} long persona conversations.")


# ============================================================
# 8) SAVE OUTPUT
# ============================================================

OUT_PATH = Path("data/persona_memory_eval_long_dataset.json")
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"💾 Saved dataset → {OUT_PATH}")
