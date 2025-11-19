import json
from pathlib import Path
import random


NUM_CONVERSATIONS = 50
OUTPUT_PATH = Path("data/memory_eval_persona_conflicts.json")

RNG = random.Random(42)


FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey",
    "Riley", "Sam", "Jamie", "Chris", "Avery"
]

CITIES_OLD = ["Paris", "London", "New York", "Berlin", "Tokyo", "Madrid"]
CITIES_NEW = ["Rome", "Toronto", "Sydney", "Chicago", "Lisbon", "Seoul"]

JOBS_OLD = [
    "software engineer",
    "teacher",
    "nurse",
    "graphic designer",
    "chef",
    "mechanical engineer",
]

JOBS_NEW = [
    "data scientist",
    "university professor",
    "product manager",
    "artist",
    "restaurant owner",
    "electrical engineer",
]

HOBBIES = [
    "playing basketball",
    "reading fantasy books",
    "hiking in the mountains",
    "playing the guitar",
    "cooking new recipes",
    "photography",
]

def sample_persona():
    """Generate a simple persona with original and updated (conflicting) facts."""
    name = RNG.choice(FIRST_NAMES)
    old_city = RNG.choice(CITIES_OLD)
    new_city = RNG.choice([c for c in CITIES_NEW if c != old_city])

    old_job = RNG.choice(JOBS_OLD)
    new_job = RNG.choice([j for j in JOBS_NEW if j != old_job])

    hobby = RNG.choice(HOBBIES)

    return {
        "name": name,
        "old_city": old_city,
        "new_city": new_city,
        "old_job": old_job,
        "new_job": new_job,
        "hobby": hobby,
    }


def build_conversation(idx: int):
    """
    Build one conversation in the same format as your friend's memory_eval_dataset:
    {
      "conversation_id": "eval_0001",
      "turns": [ ... ]
    }
    """

    persona = sample_persona()
    name = persona["name"]
    old_city = persona["old_city"]
    new_city = persona["new_city"]
    old_job = persona["old_job"]
    new_job = persona["new_job"]
    hobby = persona["hobby"]

    turns = []
    t = 1

    turns.append({
        "turn": t,
        "type": "filler",
        "prompt": (
            f"Hi, I'm {name}. I live in {old_city}, I work as a {old_job}, "
            f"and I really enjoy {hobby}. Nice to meet you!"
        )
    })
    t += 1

    turns.append({
        "turn": t,
        "type": "filler",
        "prompt": (
            f"I often spend weekends in {old_city} {hobby}. "
            f"It helps me relax after my work as a {old_job}."
        )
    })
    t += 1

    turns.append({
        "turn": t,
        "type": "filler",
        "prompt": (
            f"Actually, I forgot to tell you: I recently moved from {old_city} to {new_city}. "
            f"So now I live in {new_city}, not {old_city} anymore."
        )
    })
    t += 1

    turns.append({
        "turn": t,
        "type": "filler",
        "prompt": (
            f"And my job changed too. I used to be a {old_job}, "
            f"but now I work as a {new_job}. "
            f"It was a big decision, but I felt it fit me better."
        )
    })
    t += 1

    turns.append({
        "turn": t,
        "type": "filler",
        "prompt": (
            f"In {new_city}, my schedule as a {new_job} is very flexible, "
            f"so I still find time for {hobby}."
        )
    })
    t += 1

    implicit_prompt = (
        f"Given everything I told you about where I live and my work, "
        f"how does that connect to the fact that I first mentioned living in {old_city} "
        f"and being a {old_job}?"
    )
    implicit_gold = (
        f"You originally said you lived in {old_city} and worked as a {old_job}, "
        f"but later you clarified that you actually moved to {new_city} "
        f"and changed your job to {new_job}."
    )
    turns.append({
        "turn": t,
        "type": "implicit_probe",
        "prompt": implicit_prompt,
        "gold_answer": implicit_gold,
        "keywords": [
            old_city, new_city, old_job, new_job, "moved", "changed job"
        ]
    })
    t += 1

    turns.append({
        "turn": t,
        "type": "filler",
        "prompt": (
            f"By the way, the move from {old_city} to {new_city} was mainly because "
            f"I found a great opportunity as a {new_job}."
        )
    })
    t += 1

    explicit_prompt = (
        f"What did I say earlier about where I live now and what my current job is?"
    )
    explicit_gold = (
        f"You said that you now live in {new_city} and that you currently work as a {new_job}."
    )
    turns.append({
        "turn": t,
        "type": "explicit_probe",
        "prompt": explicit_prompt,
        "gold_answer": explicit_gold,
        "keywords": [
            new_city, new_job, "live", "currently"
        ]
    })
    t += 1

    turns.append({
        "turn": t,
        "type": "filler",
        "prompt": (
            f"So yeah, {new_city} as a {new_job} who loves {hobby} has been a big change for me."
        )
    })

    conv_id = f"eval_{idx:04d}"
    return {
        "conversation_id": conv_id,
        "turns": turns
    }


def main():
    conversations = []
    for i in range(1, NUM_CONVERSATIONS + 1):
        conv = build_conversation(i)
        conversations.append(conv)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(conversations)} conversations → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
