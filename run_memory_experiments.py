"""
run_memory_experiments.py

Runs Gemini on the data files with multiple memory methods:
  - no_memory
  - sliding_window
  - retrieval
  - hybrid

For each probe turn (explicit or implicit), it:
  - builds a context according to the memory method
  - sends the probe prompt + context to Gemini
  - saves model answers + metadata

Output:
  results files   (one JSON object per probe)

Requirements:
  pip install google-generativeai sentence-transformers numpy

Env:
  export GOOGLE_API_KEY="YOUR_KEY"
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer

# Try to import Gemini SDK; if missing or no key, we'll use a dummy.
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# -----------------------------
# Config
# -----------------------------

DATA_PATH = Path("C:\\Users\\LENOVO\\PycharmProjects\\DataLab2\\data\\memory_eval_dataset_quality.json")
RESULTS_DIR = Path("results")
RESULTS_PATH = RESULTS_DIR / "gemini_memory_runs3.jsonl"

GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Memory methods to run
MEMORY_METHODS = ["no_memory", "sliding_window", "retrieval", "hybrid"]

# Sliding window size (number of previous filler turns)
SLIDING_WINDOW_SIZE = 6

# Retrieval settings
RETRIEVAL_TOP_K = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Utility: Load dataset
# -----------------------------

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Embedding / Retrieval
# -----------------------------

class SimpleRetriever:
    """
      - For each probe, we embed persona facts + previous filler prompts
      - We embed the probe prompt and pick top_k by cosine similarity
    We used numpy and sentence-transformers.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)  # dimension for MiniLM
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb

    def top_k(self, query: str, candidates: List[str], k: int) -> List[str]:
        if not candidates:
            return []
        cand_emb = self.embed(candidates)
        query_emb = self.embed([query])[0]
        # cosine similarity
        denom = (np.linalg.norm(cand_emb, axis=1) * np.linalg.norm(query_emb) + 1e-8)
        sims = cand_emb @ query_emb / denom
        idx_sorted = np.argsort(-sims)[:k]
        return [candidates[i] for i in idx_sorted]


# -----------------------------
# Gemini wrapper
# -----------------------------

def setup_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not (HAS_GEMINI and api_key):
        print("⚠ Gemini not available (no SDK or no GOOGLE_API_KEY). Using dummy model outputs.")
        return None

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"🤖 Gemini model configured: {GEMINI_MODEL_NAME}")
        return model
    except Exception as e:
        print(f"⚠ Failed to initialize Gemini: {e}")
        return None


def call_gemini(model, user_prompt: str) -> str:
    """
    Call Gemini if available; otherwise return a dummy answer.
    """
    if model is None:
        # Fallback: echo-style dummy answer for debugging
        return f"[DUMMY ANSWER] {user_prompt[:150]}"

    try:
        response = model.generate_content(user_prompt)
        # safety: if empty or no text parts
        if not response or not getattr(response, "text", "").strip():
            return "[EMPTY GEMINI RESPONSE]"
        return response.text.strip()
    except Exception as e:
        print(f"⚠ Gemini error: {e}")
        return "[GEMINI ERROR]"


# -----------------------------
# Memory method abstraction
# -----------------------------

@dataclass
class MemoryContext:
    method: str
    context_str: str


class MemoryMethodBase:
    name: str

    def build_context(
        self,
        persona: List[str],
        turns: List[Dict[str, Any]],
        current_turn: int,
        probe_prompt: str,
        retriever: Optional[SimpleRetriever] = None
    ) -> MemoryContext:
        raise NotImplementedError


class NoMemory(MemoryMethodBase):
    name = "no_memory"

    def build_context(
        self,
        persona,
        turns,
        current_turn,
        probe_prompt,
        retriever=None
    ) -> MemoryContext:
        # No history at all: context is empty string
        return MemoryContext(method=self.name, context_str="")


class SlidingWindowMemory(MemoryMethodBase):
    name = "sliding_window"

    def __init__(self, window_size: int = SLIDING_WINDOW_SIZE):
        self.window_size = window_size

    def build_context(
        self,
        persona,
        turns,
        current_turn,
        probe_prompt,
        retriever=None
    ) -> MemoryContext:
        # Use persona + last N filler turns before current_turn
        persona_block = ""
        if persona:
            persona_block = "Here are some facts about me:\n" + "\n".join(
                f"- {p}" for p in persona
            )

        filler_before = [
            t["prompt"] for t in turns
            if t["turn"] < current_turn and t.get("type") == "filler"
        ]
        filler_window = filler_before[-self.window_size:]

        history_block = ""
        if filler_window:
            history_block = "\n\nSome things I said earlier:\n" + "\n".join(
                f"- {u}" for u in filler_window
            )

        total = (persona_block + history_block).strip()
        return MemoryContext(method=self.name, context_str=total)


class RetrievalMemory(MemoryMethodBase):
    name = "retrieval"

    def build_context(
        self,
        persona,
        turns,
        current_turn,
        probe_prompt,
        retriever: Optional[SimpleRetriever] = None
    ) -> MemoryContext:
        if retriever is None:
            return MemoryContext(method=self.name, context_str="")

        persona_items = persona or []
        filler_before = [
            t["prompt"]
            for t in turns
            if t["turn"] < current_turn and t.get("type") == "filler"
        ]
        candidates = persona_items + filler_before
        top_k = retriever.top_k(probe_prompt, candidates, RETRIEVAL_TOP_K)

        block = ""
        if top_k:
            block = "Relevant things I said earlier or facts about me:\n" + "\n".join(
                f"- {item}" for item in top_k
            )

        return MemoryContext(method=self.name, context_str=block)


class HybridMemory(MemoryMethodBase):
    name = "hybrid"

    def __init__(self, window_size: int = SLIDING_WINDOW_SIZE):
        self.window_size = window_size

    def build_context(
        self,
        persona,
        turns,
        current_turn,
        probe_prompt,
        retriever: Optional[SimpleRetriever] = None
    ) -> MemoryContext:
        # retrieval + small sliding history
        if retriever is None:
            retriever_block = ""
        else:
            persona_items = persona or []
            filler_before = [
                t["prompt"]
                for t in turns
                if t["turn"] < current_turn and t.get("type") == "filler"
            ]
            candidates = persona_items + filler_before
            retrieved = retriever.top_k(probe_prompt, candidates, RETRIEVAL_TOP_K)
            if retrieved:
                retriever_block = "Important earlier info:\n" + "\n".join(
                    f"- {r}" for r in retrieved
                )
            else:
                retriever_block = ""

        filler_before = [
            t["prompt"] for t in turns
            if t["turn"] < current_turn and t.get("type") == "filler"
        ]
        filler_window = filler_before[-self.window_size:]
        window_block = ""
        if filler_window:
            window_block = "\n\nRecent things I said:\n" + "\n".join(
                f"- {u}" for u in filler_window
            )

        persona_block = ""
        if persona:
            persona_block = "Here are some facts about me:\n" + "\n".join(
                f"- {p}" for p in persona
            ) + "\n\n"

        total = (persona_block + retriever_block + window_block).strip()
        return MemoryContext(method=self.name, context_str=total)


# -----------------------------
# Main loop
# -----------------------------

def get_memory_method_instance(name: str) -> MemoryMethodBase:
    if name == "no_memory":
        return NoMemory()
    if name == "sliding_window":
        return SlidingWindowMemory()
    if name == "retrieval":
        return RetrievalMemory()
    if name == "hybrid":
        return HybridMemory()
    raise ValueError(f"Unknown memory method: {name}")


def is_probe_turn(turn_obj: Dict[str, Any]) -> bool:
    t = turn_obj.get("type", "")
    return t in ("explicit_probe", "implicit_probe")


def build_user_prompt(context: str, probe_prompt: str, method_name: str) -> str:
    """
    Build the single text prompt we send to Gemini.
    For no_memory: just the probe question.
    For others: context + probe.
    """
    if method_name == "no_memory" or not context.strip():
        return probe_prompt.strip()

    return (
        f"{context.strip()}\n\n"
        f"Now here is my question:\n"
        f"{probe_prompt.strip()}"
    )


def main():
    ensure_results_dir()
    conversations = load_dataset(DATA_PATH)
    model = setup_gemini()
    retriever = SimpleRetriever()

    total_probes = 0
    for conv in conversations:
        for t in conv.get("turns", []):
            if is_probe_turn(t):
                total_probes += 1
    print(f"📂 Loaded {len(conversations)} conversations with {total_probes} probes total.")

    # Open output file in write mode (overwrite each run)
    with RESULTS_PATH.open("w", encoding="utf-8") as out_f:
        # For each memory method
        for method_name in MEMORY_METHODS:
            mem_method = get_memory_method_instance(method_name)
            print(f"\n🚀 Running memory method: {method_name}")

            # Iterate over all conversations
            for conv in conversations:
                conv_id = conv.get("conversation_id", "unknown_id")
                persona = conv.get("persona", [])
                turns = conv.get("turns", [])

                # For each probe turn
                for turn_obj in turns:
                    if not is_probe_turn(turn_obj):
                        continue

                    turn_idx = turn_obj["turn"]
                    probe_type = turn_obj.get("type")
                    probe_prompt = turn_obj.get("prompt", "")
                    related_fact = turn_obj.get("related_fact", "")
                    gold_answer = turn_obj.get("gold_answer", "")
                    keywords = turn_obj.get("keywords", [])

                    # Build context according to memory method
                    mem_ctx = mem_method.build_context(
                        persona=persona,
                        turns=turns,
                        current_turn=turn_idx,
                        probe_prompt=probe_prompt,
                        retriever=retriever
                    )

                    user_prompt = build_user_prompt(
                        context=mem_ctx.context_str,
                        probe_prompt=probe_prompt,
                        method_name=method_name
                    )

                    model_answer = call_gemini(model, user_prompt)

                    record = {
                        "conversation_id": conv_id,
                        "memory_method": method_name,
                        "probe_turn": turn_idx,
                        "probe_type": probe_type,
                        "probe_prompt": probe_prompt,
                        "related_fact": related_fact,
                        "gold_answer": gold_answer,
                        "keywords": keywords,
                        "context_used": mem_ctx.context_str,
                        "model_answer": model_answer,
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"✅ Finished method: {method_name}")

    print(f"\n📁 All results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

