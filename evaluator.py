# # evaluator.py
# import json
# from typing import List, Dict
# from collections import defaultdict
# import google.generativeai as genai
# from config import *
#
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel(MODEL_NAME)
#
#
# def evaluate_fact_recall_keyword(prediction: str, fact: str) -> float:
#     """Simple keyword-based evaluation"""
#     pred_lower = prediction.lower()
#     fact_lower = fact.lower()
#
#     # Extract key terms from fact
#     fact_words = set(fact_lower.split())
#     # Remove common words
#     stop_words = {'i', 'am', 'is', 'the', 'a', 'an', 'to', 'my', 'as'}
#     key_words = fact_words - stop_words
#
#     # Check how many key words appear in prediction
#     matches = sum(1 for word in key_words if word in pred_lower)
#
#     return matches / len(key_words) if key_words else 0.0
#
#
# def evaluate_with_llm(question: str, prediction: str, gold_answer: str, fact: str) -> Dict:
#     """Use LLM to evaluate if prediction is correct"""
#     prompt = f"""Evaluate if the predicted answer is correct given the fact.
#
# Fact: {fact}
# Question: {question}
# Gold Answer: {gold_answer}
# Predicted Answer: {prediction}
#
# Does the predicted answer correctly use/mention the fact?
# Answer with just: CORRECT, PARTIALLY_CORRECT, or INCORRECT
#
# Then on a new line, briefly explain why (one sentence)."""
#
#     try:
#         response = model.generate_content(prompt)
#         text = response.text.strip()
#         lines = text.split('\n')
#
#         verdict = lines[0].strip().upper()
#         explanation = lines[1] if len(lines) > 1 else ""
#
#         # Convert to score
#         if "CORRECT" in verdict and "PARTIALLY" not in verdict:
#             score = 1.0
#         elif "PARTIALLY" in verdict:
#             score = 0.5
#         else:
#             score = 0.0
#
#         return {
#             "score": score,
#             "verdict": verdict,
#             "explanation": explanation
#         }
#     except Exception as e:
#         print(f"LLM evaluation error: {e}")
#         # Fallback to keyword matching
#         keyword_score = evaluate_fact_recall_keyword(prediction, fact)
#         return {
#             "score": keyword_score,
#             "verdict": "KEYWORD_FALLBACK",
#             "explanation": f"Keyword match: {keyword_score:.2f}"
#         }
#
#
# def evaluate_all():
#     """Evaluate all predictions"""
#     print("\n" + "=" * 50)
#     print("EVALUATION")
#     print("=" * 50)
#
#     # Load predictions
#     with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
#         predictions = json.load(f)
#
#     # Evaluate each prediction
#     evaluated = []
#     for pred in predictions:
#         print(f"\nEvaluating Conv {pred['conversation_id']}, Turn {pred['turn']}...")
#
#         eval_result = evaluate_with_llm(
#             pred["question"],
#             pred["prediction"],
#             pred["gold_answer"],
#             pred["related_fact"]
#         )
#
#         pred["eval_score"] = eval_result["score"]
#         pred["eval_verdict"] = eval_result["verdict"]
#         pred["eval_explanation"] = eval_result["explanation"]
#
#         evaluated.append(pred)
#
#     # Calculate metrics by system
#     results_by_system = defaultdict(lambda: {
#         "total": 0,
#         "correct": 0,
#         "scores": [],
#         "by_type": {"explicit": [], "implicit": []}
#     })
#
#     for pred in evaluated:
#         system = pred["system"]
#         score = pred["eval_score"]
#         probe_type = pred["probe_type"]
#
#         results_by_system[system]["total"] += 1
#         results_by_system[system]["scores"].append(score)
#         results_by_system[system]["by_type"][probe_type].append(score)
#
#         if score >= 0.5:  # Count partial as success
#             results_by_system[system]["correct"] += 1
#
#     # Compute final metrics
#     final_results = {}
#     for system, data in results_by_system.items():
#         accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
#         avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
#
#         explicit_scores = data["by_type"]["explicit"]
#         implicit_scores = data["by_type"]["implicit"]
#
#         final_results[system] = {
#             "accuracy": round(accuracy * 100, 2),
#             "average_score": round(avg_score, 3),
#             "total_questions": data["total"],
#             "correct_answers": data["correct"],
#             "explicit_accuracy": round(sum(explicit_scores) / len(explicit_scores) * 100, 2) if explicit_scores else 0,
#             "implicit_accuracy": round(sum(implicit_scores) / len(implicit_scores) * 100, 2) if implicit_scores else 0
#         }
#
#     # Print results
#     print("\n" + "=" * 50)
#     print("RESULTS")
#     print("=" * 50)
#     for system, metrics in final_results.items():
#         print(f"\n{system.upper()}:")
#         print(f"  Overall Accuracy: {metrics['accuracy']}%")
#         print(f"  Average Score: {metrics['average_score']}")
#         print(f"  Explicit Questions: {metrics['explicit_accuracy']}%")
#         print(f"  Implicit Questions: {metrics['implicit_accuracy']}%")
#         print(f"  Correct: {metrics['correct_answers']}/{metrics['total_questions']}")
#
#     # Save results
#     output = {
#         "predictions": evaluated,
#         "summary": final_results
#     }
#
#     with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
#         json.dump(output, f, indent=2, ensure_ascii=False)
#
#     print(f"\n✅ Results saved to: {RESULTS_FILE}")
#
#
# if __name__ == "__main__":
#     evaluate_all()
# evaluator.py
import json
from difflib import SequenceMatcher
from config import *


def string_similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def evaluate_fact_recall_keyword(prediction: str, fact: str) -> dict:
    """Check if fact keywords appear in prediction"""
    pred_lower = prediction.lower()
    fact_lower = fact.lower()

    fact_words = set(fact_lower.split())
    stop_words = {'i', 'am', 'is', 'the', 'a', 'an', 'to', 'my', 'as', 'at', 'in', 'on'}
    key_words = fact_words - stop_words

    matched = [w for w in key_words if w in pred_lower]
    missing = [w for w in key_words if w not in pred_lower]

    score = len(matched) / len(key_words) if key_words else 0.0

    return {
        "score": score,
        "matched": matched,
        "missing": missing
    }


def evaluate_prediction(pred: dict) -> dict:
    """Evaluate one prediction using similarity"""
    prediction = pred["prediction"]
    gold_answer = pred["gold_answer"]
    fact = pred["related_fact"]

    # 1. Text similarity to gold answer
    gold_sim = string_similarity(prediction, gold_answer)

    # 2. Keyword matching with fact
    keyword_result = evaluate_fact_recall_keyword(prediction, fact)

    # 3. Combined score
    combined = (0.6 * gold_sim) + (0.4 * keyword_result["score"])

    # Verdict
    if combined >= 0.7:
        verdict = "CORRECT"
    elif combined >= 0.4:
        verdict = "PARTIAL"
    else:
        verdict = "INCORRECT"

    return {
        "score": combined,
        "gold_similarity": gold_sim,
        "keyword_score": keyword_result["score"],
        "verdict": verdict,
        "matched_keywords": keyword_result["matched"],
        "missing_keywords": keyword_result["missing"]
    }


def evaluate_all():
    """Evaluate baseline predictions"""
    print("\n" + "=" * 70)
    print("📊 EVALUATION")
    print("=" * 70)

    # Load predictions
    with open(PREDICTIONS_FILE, 'r') as f:
        predictions = json.load(f)

    print(f"Total predictions: {len(predictions)}")

    # Evaluate each
    results = []
    correct = 0
    by_type = {"explicit": [], "implicit": []}
    by_topic = {}

    for i, pred in enumerate(predictions, 1):
        eval_result = evaluate_prediction(pred)

        print(f"\n[{i}/{len(predictions)}] Conv {pred['conversation_id']}, Turn {pred['turn']}")
        print(f"  ❓ {pred['question'][:60]}...")
        print(f"  📌 Fact: {pred['related_fact']}")
        print(f"  🤖 Prediction: {pred['prediction'][:60]}...")
        print(f"  ✅ Gold: {pred['gold_answer'][:60]}...")
        print(f"  📊 Score: {eval_result['score']:.3f} ({eval_result['verdict']})")
        print(f"     Gold similarity: {eval_result['gold_similarity']:.3f}")
        print(f"     Keyword match: {eval_result['keyword_score']:.3f}")
        if eval_result['matched_keywords']:
            print(f"     ✓ Matched: {', '.join(eval_result['matched_keywords'])}")
        if eval_result['missing_keywords']:
            print(f"     ✗ Missing: {', '.join(eval_result['missing_keywords'])}")

        # Store result
        pred.update(eval_result)
        results.append(pred)

        # Aggregate stats
        if eval_result["score"] >= 0.5:
            correct += 1

        probe_type = pred["probe_type"]
        by_type[probe_type].append(eval_result["score"])

        topic = pred["topic"]
        if topic not in by_topic:
            by_topic[topic] = []
        by_topic[topic].append(eval_result["score"])

    # Summary
    accuracy = (correct / len(predictions)) * 100 if predictions else 0
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    print("\n" + "=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print(f"Average Score: {avg_score:.3f}")
    print(f"Correct (≥50%): {correct}/{len(predictions)}")

    print(f"\nBy Question Type:")
    for qtype, scores in by_type.items():
        if scores:
            avg = sum(scores) / len(scores) * 100
            print(f"  {qtype}: {avg:.1f}% ({len(scores)} questions)")

    print(f"\nBy Topic:")
    for topic, scores in by_topic.items():
        if scores:
            avg = sum(scores) / len(scores) * 100
            print(f"  {topic}: {avg:.1f}% ({len(scores)} questions)")

    # Save
    output = {
        "predictions": results,
        "summary": {
            "accuracy": round(accuracy, 2),
            "average_score": round(avg_score, 3),
            "total": len(predictions),
            "correct": correct,
            "by_type": {k: round(sum(v) / len(v) * 100, 2) if v else 0
                        for k, v in by_type.items()},
            "by_topic": {k: round(sum(v) / len(v) * 100, 2) if v else 0
                         for k, v in by_topic.items()}
        }
    }

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    evaluate_all()