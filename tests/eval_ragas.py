import sys
import types
import warnings

# Suppress RAGAS deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create a fake VertexAI module so RAGAS doesn't crash on startup
dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_vertex.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex

import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.models import RagEngine


def run_test():
    engine = RagEngine()

    # Clear layer 1 cache, so that every answer is given with context, not just cached answer
    if hasattr(engine, "clear_all_caches"):
        print(" Clearing Layer 1 Cache for clean evaluation...")
        engine.clear_all_caches()

    # Setup Groq Evaluator (8B model for higher daily rate limits)
    evaluator_llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        max_retries=10
    )

    evaluator_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Override Groq n > 1 Restriction
    answer_relevancy.strictness = 1

    # Prevent concurrent rate spiking
    run_config = RunConfig(
        max_workers=1,
        timeout=180
    )

    # load Test Dataset
    with open("eval_set.json", "r") as f:
        test_cases = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    print(" Querying RagEngine...")
    for i, case in enumerate(test_cases, 1):
        q = case["question"]
        gt = case["ground_truth"]

        print(f"\n[{i}/{len(test_cases)}] Processing: {q}")

        try:
            res = engine.search_and_generate(
                question=q,
                allowed_clearances=["public"]
            )
        except Exception as e:
            print(f" Exception in search_and_generate: {e}")
            res = None

        # Safe Unpacking Guard
        if res is None:
            print(f"⚠ Warning: engine.search_and_generate() returned None for query: '{q}'")
            answer = "No response generated due to an internal engine error."
            citations = []
            retrieved_chunks = []
        else:
            answer, citations, retrieved_chunks = res

        chunk_texts = [
            c["text"] if isinstance(c, dict) and "text" in c else str(c)
            for c in retrieved_chunks
        ]

        questions.append(q)
        answers.append(answer)
        contexts.append(chunk_texts)
        ground_truths.append(gt)

    #  Run RAGAS Benchmarking
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(eval_data)

    print("\n📊 Evaluating outputs with RAGAS + Groq (llama-3.1-8b-instant)...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    # Output and Save results ---
    print("\n=== REAL RAGAS BENCHMARK RESULTS ===")
    print(results)

    df = results.to_pandas()
    df.to_csv("real_ragas_results.csv", index=False)
    print("\nSaved output to real_ragas_results.csv!")


if __name__ == "__main__":
    run_test()