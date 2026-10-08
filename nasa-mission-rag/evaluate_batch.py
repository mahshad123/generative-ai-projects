"""Batch RAGAS evaluation over a test set of NASA mission questions.
"""

import os
import json
from statistics import mean

import rag_client
import llm_client
import ragas_evaluator

CHROMA_DIR = "./chroma_db_openai"
COLLECTION_NAME = "nasa_space_missions_text"
DATASET_FILE = "test_questions.json"
RESULTS_FILE = "evaluation_results.json"
N_RESULTS = 3
MODEL = "gpt-3.5-turbo"


def load_questions(path: str):
    """Load the evaluation dataset, with a clear error if it's malformed."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Dataset file not found: {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Dataset file is not valid JSON: {e}")
        return []

    if not isinstance(data, list) or not data:
        print("Dataset must be a non-empty JSON list of {category, question}.")
        return []
    return data


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OPENAI_API_KEY found. Set it first:  export OPENAI_API_KEY=\"sk-...\"")
        return
    os.environ.setdefault("CHROMA_OPENAI_API_KEY", api_key)

    questions = load_questions(DATASET_FILE)
    if not questions:
        return

    try:
        collection = rag_client.initialize_rag_system(CHROMA_DIR, COLLECTION_NAME)
    except Exception as e:
        print(f"Could not open ChromaDB collection '{COLLECTION_NAME}' in "
              f"'{CHROMA_DIR}': {e}")
        print("Run the embedding pipeline first to build the database.")
        return

    results = []
    metric_values = {}

    for i, item in enumerate(questions, 1):
        question = item.get("question", "")
        category = item.get("category", "uncategorized")

        docs = rag_client.retrieve_documents(collection, question, n_results=N_RESULTS)
        if docs and docs.get("documents"):
            contexts = docs["documents"][0]
            metadatas = docs["metadatas"][0]
            context = rag_client.format_context(contexts, metadatas)
        else:
            contexts = []
            context = ""

        answer = llm_client.generate_response(api_key, question, context, [], MODEL)

        scores = ragas_evaluator.evaluate_response_quality(question, answer, contexts)

        print("=" * 70)
        print(f"Q{i} [{category}]: {question}")
        print(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        if "error" in scores:
            print(f"Evaluation error: {scores['error']}")
        else:
            print(f"Scores: {scores}")
            for metric_name, value in scores.items():
                if isinstance(value, (int, float)):
                    metric_values.setdefault(metric_name, []).append(value)

        results.append({
            "category": category,
            "question": question,
            "answer": answer,
            "scores": scores,
        })

    print("=" * 70)
    print(f"AGGREGATE (mean per metric over {len(questions)} questions):")
    aggregate = {}
    if metric_values:
        for metric_name, values in metric_values.items():
            avg = round(mean(values), 4)
            aggregate[metric_name] = avg
            print(f"  {metric_name}: {avg}  (from {len(values)} scored answers)")
    else:
        print("  No numeric metrics were produced (see per-question errors above).")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": results, "aggregate": aggregate}, f, indent=2)
    print("=" * 70)
    print(f"Full results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
