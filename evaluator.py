# evaluator.py
import json
from typing import List, Dict
from collections import defaultdict
import google.generativeai as genai
from config import *

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


def evaluate_fact_recall_keyword(prediction: str, fact: str) -> float:
    """Simple keyword-based evaluation"""
    pred_lower = prediction.lower()
    fact_lower = fact.lower()

    # Extract key terms from fact
    fact_words = set(fact_lower.split())
    # Remove common words
    stop_words = {'i', 'am', 'is', 'the', 'a', 'an', 'to', 'my', 'as'}
    key_words = fact_words - stop_words

    # Check how many key words appear in prediction
    matches = sum(1 for word in key_words if word in pred_lower)

    return matches / len(key_words) if key_words else 0.0


def evaluate_with_llm(question: str, prediction: str, gold_answer: str, fact: str) -> Dict:
    """Use LLM to evaluate if prediction is correct"""
    prompt = f"""Evaluate if the predicted answer is correct given the fact.

Fact: {fact}
Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {prediction}

Does the predicted answer correctly use/mention the fact? 
Answer with just: CORRECT, PARTIALLY_CORRECT, or INCORRECT

Then on a new line, briefly explain why (one sentence)."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        lines = text.split('\n')

        verdict = lines[0].strip().upper()
        explanation = lines[1] if len(lines) > 1 else ""

        # Convert to score
        if "CORRECT" in verdict and "PARTIALLY" not in verdict:
            score = 1.0
        elif "PARTIALLY" in verdict:
            score = 0.5
        else:
            score = 0.0

        return {
            "score": score,
            "verdict": verdict,
            "explanation": explanation
        }
    except Exception as e:
        print(f"LLM evaluation error: {e}")
        # Fallback to keyword matching
        keyword_score = evaluate_fact_recall_keyword(prediction, fact)
        return {
            "score": keyword_score,
            "verdict": "KEYWORD_FALLBACK",
            "explanation": f"Keyword match: {keyword_score:.2f}"
        }


def evaluate_all():
    """Evaluate all predictions"""
    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    # Load predictions
    with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # Evaluate each prediction
    evaluated = []
    for pred in predictions:
        print(f"\nEvaluating Conv {pred['conversation_id']}, Turn {pred['turn']}...")

        eval_result = evaluate_with_llm(
            pred["question"],
            pred["prediction"],
            pred["gold_answer"],
            pred["related_fact"]
        )

        pred["eval_score"] = eval_result["score"]
        pred["eval_verdict"] = eval_result["verdict"]
        pred["eval_explanation"] = eval_result["explanation"]

        evaluated.append(pred)

    # Calculate metrics by system
    results_by_system = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "scores": [],
        "by_type": {"explicit": [], "implicit": []}
    })

    for pred in evaluated:
        system = pred["system"]
        score = pred["eval_score"]
        probe_type = pred["probe_type"]

        results_by_system[system]["total"] += 1
        results_by_system[system]["scores"].append(score)
        results_by_system[system]["by_type"][probe_type].append(score)

        if score >= 0.5:  # Count partial as success
            results_by_system[system]["correct"] += 1

    # Compute final metrics
    final_results = {}
    for system, data in results_by_system.items():
        accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0

        explicit_scores = data["by_type"]["explicit"]
        implicit_scores = data["by_type"]["implicit"]

        final_results[system] = {
            "accuracy": round(accuracy * 100, 2),
            "average_score": round(avg_score, 3),
            "total_questions": data["total"],
            "correct_answers": data["correct"],
            "explicit_accuracy": round(sum(explicit_scores) / len(explicit_scores) * 100, 2) if explicit_scores else 0,
            "implicit_accuracy": round(sum(implicit_scores) / len(implicit_scores) * 100, 2) if implicit_scores else 0
        }

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for system, metrics in final_results.items():
        print(f"\n{system.upper()}:")
        print(f"  Overall Accuracy: {metrics['accuracy']}%")
        print(f"  Average Score: {metrics['average_score']}")
        print(f"  Explicit Questions: {metrics['explicit_accuracy']}%")
        print(f"  Implicit Questions: {metrics['implicit_accuracy']}%")
        print(f"  Correct: {metrics['correct_answers']}/{metrics['total_questions']}")

    # Save results
    output = {
        "predictions": evaluated,
        "summary": final_results
    }

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    evaluate_all()