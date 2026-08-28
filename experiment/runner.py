import json
import os
import time

from .config import (
    MAX_QUESTIONS,
    RESULTS_DIR
)

from .plain_llm import PlainLLM
from .rag_system import HybridRAGSystem


def load_dataset(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_results(
    results,
    filename
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


def run_experiment(
    dataset_path
):

    dataset = load_dataset(
        dataset_path
    )

    dataset = dataset[
        :MAX_QUESTIONS
    ]

    print(
        f"Running Experiment A "
        f"with {len(dataset)} questions."
    )

    plain_llm = PlainLLM()

    rag_system = (
        HybridRAGSystem()
    )

    # --------------------------------------------------------
    # Clear semantic cache BEFORE experiment.
    # --------------------------------------------------------

    try:

        rag_system.rag.clear_all_caches()

        print(
            "[RAG] Semantic caches cleared."
        )

    except AttributeError:

        print(
            "[RAG] clear_all_caches() "
            "not available. Continuing."
        )

    results = []

    for index, item in enumerate(
        dataset,
        start=1
    ):

        question_id = item["id"]

        question = item["question"]

        print()
        print("=" * 80)

        print(
            f"QUESTION {index}/"
            f"{len(dataset)}"
        )

        print(
            f"ID: {question_id}"
        )

        print(
            f"Question: {question}"
        )

        print("=" * 80)

        # ====================================================
        # PLAIN LLM
        # ====================================================

        print()
        print("[1/2] Running Plain LLM...")

        plain_start = (
            time.perf_counter()
        )

        plain_result = (
            plain_llm.generate(
                question
            )
        )

        plain_wall_latency = (
            time.perf_counter()
            - plain_start
        )

        print()
        print("[PLAIN LLM ANSWER]")
        print(
            plain_result["answer"]
        )

        print(
            f"Tokens: "
            f"{plain_result['total_tokens']}"
        )

        print(
            f"Latency: "
            f"{plain_wall_latency:.2f}s"
        )

        # ====================================================
        # HYBRID RAG
        # ====================================================

        print()
        print("[2/2] Running Hybrid RAG...")

        rag_start = (
            time.perf_counter()
        )

        rag_result = (
            rag_system.generate(
                question
            )
        )

        rag_wall_latency = (
            time.perf_counter()
            - rag_start
        )

        print()
        print("[HYBRID RAG ANSWER]")

        print(
            rag_result["answer"]
        )

        print()
        print("[RAG CITATIONS]")

        print(
            rag_result["citations"]
        )

        print()
        print(
            f"RAG latency: "
            f"{rag_wall_latency:.2f}s"
        )

        # ====================================================
        # STORE
        # ====================================================

        result = {

            "id":
                question_id,

            "category":
                item.get(
                    "category"
                ),

            "question":
                question,

            "ground_truth": {

                "expected_answer":
                    item.get(
                        "expected_answer",
                        ""
                    ),

                "source":
                    item.get(
                        "source",
                        ""
                    ),

                "page":
                    item.get(
                        "page"
                    )
            },

            "plain_llm": {

                "answer":
                    plain_result[
                        "answer"
                    ],

                "latency_seconds":
                    plain_wall_latency,

                "prompt_tokens":
                    plain_result[
                        "prompt_tokens"
                    ],

                "completion_tokens":
                    plain_result[
                        "completion_tokens"
                    ],

                "total_tokens":
                    plain_result[
                        "total_tokens"
                    ],

                "rate_limits":
                    plain_result[
                        "rate_limits"
                    ]
            },

            "hybrid_rag": {

                "answer":
                    rag_result[
                        "answer"
                    ],

                "citations":
                    rag_result[
                        "citations"
                    ],

                "retrieved_chunks":
                    rag_result[
                        "retrieved_chunks"
                    ],

                "latency_seconds":
                    rag_wall_latency
            }
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Save after EVERY question.
        #
        # This is important.
        #
        # If Groq stops us after question 37,
        # questions 1-37 are already saved.
        # ----------------------------------------------------

        save_results(
            results,
            "experiment_a_raw_results.json"
        )

    print()
    print("=" * 80)
    print("EXPERIMENT A COMPLETE")
    print("=" * 80)

    return results