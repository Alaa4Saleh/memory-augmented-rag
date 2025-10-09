# fact_generator.py
import random
from typing import List, Dict

FACT_TEMPLATES = {
    "dietary": [
        "I'm vegetarian",
        "I'm vegan",
        "I'm allergic to peanuts",
        "I'm allergic to shellfish",
        "I'm lactose intolerant",
        "I can't eat gluten",
        "I don't eat spicy food",
        "I love Italian food",
        "I hate mushrooms"
    ],
    "work": [
        "I work as a software engineer",
        "I work as a data scientist",
        "I'm a teacher",
        "I work remotely",
        "I work at a startup",
        "I work night shifts",
        "I'm studying computer science at Technion",
        "I'm working on a machine learning project",
        "I manage a team of 5 people"
    ],
    "hobbies": [
        "I enjoy hiking",
        "I play guitar",
        "I love reading sci-fi novels",
        "I play tennis every weekend",
        "I'm learning Spanish",
        "I practice yoga",
        "I collect vintage records",
        "I enjoy photography",
        "I love gaming"
    ]
}


def generate_facts(num_facts: int) -> List[Dict]:
    """Generate random facts from different topics"""
    all_facts = []
    for topic, facts_list in FACT_TEMPLATES.items():
        for fact in facts_list:
            all_facts.append({"topic": topic, "fact": fact})

    # Sample without replacement
    selected = random.sample(all_facts, min(num_facts, len(all_facts)))
    return selected


def fact_to_establishment_turn(fact_obj: Dict, turn_num: int) -> Dict:
    """Convert fact to a conversation turn where user states it"""
    return {
        "turn": turn_num,
        "speaker": "user",
        "text": fact_obj["fact"],
        "type": "fact_establishment",
        "topic": fact_obj["topic"]
    }


if __name__ == "__main__":
    # Test
    facts = generate_facts(3)
    print("Generated facts:")
    for f in facts:
        print(f"  [{f['topic']}] {f['fact']}")