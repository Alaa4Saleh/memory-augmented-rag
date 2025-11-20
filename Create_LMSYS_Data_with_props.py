"""
============================================================
  MEMORY PROBE DATASET GENERATOR — QUALITY FILTERED VERSION
               (STABLE GEMINI FIXED EDITION)
============================================================

Fixes included:
✔ Stable Gemini text extraction (no response.text)
✔ No `< >` markers in prompts (avoids empty responses)
✔ Simpler structured instructions
✔ Fallback gold answers for safety
✔ Identical functionality to your original script
============================================================
"""

import os
import json
import random
import time
import re
from pathlib import Path

from dotenv import load_dotenv
from langdetect import detect
import google.generativeai as genai
from tqdm import tqdm


# ============================================================
# 1) CONFIG & API
# ============================================================

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ Missing GOOGLE_API_KEY in .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

RAW_PATH = Path("data/lmsys_conversations_raw.json")
OUT_PATH = Path("data/memory_eval_dataset_quality.json")

MAX_CONVS = 50
TURNS_PER_CONV = 20
PROBE_EVERY = 5
MIN_USER_TURNS = 12
MIN_MEANINGFUL_RATIO = 0.7
MIN_AVG_CHARS = 25

BAD_WORDS = {
    "fuck", "shit", "bitch", "asshole", "bastard", "slut",
    "whore", "dick", "cock", "pussy", "nigger", "faggot",
    "rape", "porn", "pornography", "kill myself", "suicide"
}


# ============================================================
# 2) LOAD RAW DATA
# ============================================================

if not RAW_PATH.exists():
    raise FileNotFoundError("❌ Missing data/lmsys_conversations_raw.json")

with open(RAW_PATH, "r", encoding="utf-8") as f:
    raw_conversations = json.load(f)

print(f"📂 Loaded {len(raw_conversations)} base conversations.")


# ============================================================
# 3) HELPER FUNCTIONS (SAFETY + QUALITY)
# ============================================================

def gemini_call(prompt: str, retries: int = 3) -> str:
    """Stable Gemini call using candidates/parts instead of .text."""
    for attempt in range(retries):
        try:
            r = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 180,
                },
            )

            # Safe text extraction
            if (
                hasattr(r, "candidates") and r.candidates and
                hasattr(r.candidates[0], "content") and
                r.candidates[0].content.parts
            ):
                part = r.candidates[0].content.parts[0]
                if hasattr(part, "text") and part.text:
                    return part.text.strip()

        except Exception as e:
            print(f"⚠ Gemini error: {e} (attempt {attempt + 1})")

        time.sleep(2 + attempt)

    return ""


def is_english(text: str, min_ratio: float = 0.8) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    total = 0
    en = 0
    for l in lines:
        try:
            if detect(l) == "en":
                en += 1
            total += 1
        except:
            continue
    return total > 0 and en / total >= min_ratio


def contains_bad_language(text: str) -> bool:
    t = text.lower()
    for b in BAD_WORDS:
        if b in t:
            return True
    return False


def is_meaningful_prompt(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 5:
        return False

    if stripped.lower() in {"ok", "k", "lol", "lmao", "idk", "?", "??", "???"}:
        return False

    tokens = re.findall(r"[A-Za-z]+", stripped)
    if len(tokens) < 3:
        return False

    if not any(len(t) >= 4 for t in tokens):
        return False

    return True


def clean_user_turns(conv):
    cleaned = []
    for line in conv:
        if not line.lower().startswith("user:"):
            continue
        text = line.split(":", 1)[1].strip()
        if contains_bad_language(text):
            return []
        if is_meaningful_prompt(text):
            cleaned.append(text)
    return cleaned


def conversation_passes_quality(user_prompts):
    if len(user_prompts) < MIN_USER_TURNS:
        return False
    avg_len = sum(len(p) for p in user_prompts) / len(user_prompts)
    if avg_len < MIN_AVG_CHARS:
        return False
    return True


# ============================================================
# 4) TOPIC EXTRACTION + PROBES
# ============================================================

def extract_topic(turns, window=4):
    recent = [t["prompt"] for t in turns[-window:]]
    combined = " ".join(recent)
    parts = [p.strip() for p in combined.split(".") if len(p.split()) > 3]
    return random.choice(parts) if parts else "what I mentioned earlier"


def make_explicit(topic):
    templates = [
        "What did I say earlier about {}?",
        "Can you remind me what I mentioned about {}?",
        "Do you remember my earlier point about {}?",
        "Could you repeat what I said before regarding {}?"
    ]
    return random.choice(templates).format(topic)


def make_implicit(topic):
    templates = [
        "Given what I said earlier about {}, does that still apply?",
        "How does what I mentioned before about {} relate to this?",
        "If you recall my earlier comment about {}, what do you think now?",
        "Would you say this matches what we discussed about {}?"
    ]
    return random.choice(templates).format(topic)


def gold_answer_prompt(context, question):
    return (
        "Below is part of a user's conversation:\n"
        f"{context}\n\n"
        f"The user now asks: {question}\n\n"
        "Please produce exactly TWO lines:\n"
        "Answer: (one factual sentence describing the correct recall)\n"
        "Keywords: word1, word2, word3, word4\n"
        "The keywords must be comma-separated.\n"
    )


# ============================================================
# 5) BUILD FINAL DATASET
# ============================================================

final_data = []

for idx, conv in enumerate(tqdm(raw_conversations, desc="Building quality dataset")):

    user_prompts = clean_user_turns(conv)
    if not user_prompts:
        continue

    if not is_english(" ".join(user_prompts)):
        continue

    if not conversation_passes_quality(user_prompts):
        continue

    turns = []
    turn_no = 1
    pointer = 0

    while len(turns) < TURNS_PER_CONV and pointer < len(user_prompts):

        # Add filler
        turns.append({
            "turn": turn_no,
            "type": "filler",
            "prompt": user_prompts[pointer]
        })
        turn_no += 1
        pointer += 1

        # Insert probe when needed
        if len(turns) % PROBE_EVERY == 0 and len(turns) < TURNS_PER_CONV:

            topic = extract_topic(turns)
            probe_type = random.choice(["explicit_probe", "implicit_probe"])
            question = make_explicit(topic) if probe_type == "explicit_probe" else make_implicit(topic)

            context = "\n".join(t["prompt"] for t in turns[-8:])
            g_prompt = gold_answer_prompt(context, question)
            g_raw = gemini_call(g_prompt)

            gold_answer = ""
            keywords = []

            for line in g_raw.splitlines():
                line_low = line.strip().lower()
                if line_low.startswith("answer:"):
                    gold_answer = line.split(":", 1)[1].strip()
                elif line_low.startswith("keywords:"):
                    keywords = [k.strip() for k in line.split(":", 1)[1].split(",") if k.strip()]

            # fallback
            if not gold_answer:
                gold_answer = "The user previously mentioned this topic in the conversation."
            if not keywords:
                keywords = ["memory", "recall", "topic"]

            turns.append({
                "turn": turn_no,
                "type": probe_type,
                "prompt": question,
                "gold_answer": gold_answer,
                "keywords": keywords
            })
            turn_no += 1

    if len(turns) >= 10:
        final_data.append({
            "conversation_id": f"eval_{idx:04d}",
            "turns": turns
        })

    if len(final_data) >= MAX_CONVS:
        break

print(f"🧠 Created {len(final_data)} QUALITY English conversations with probes.")


# ============================================================
# 6) SAVE
# ============================================================

OUT_PATH.parent.mkdir(exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"💾 Saved → {OUT_PATH}")
