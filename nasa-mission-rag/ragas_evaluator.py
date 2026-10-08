import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# Optional custom endpoint (e.g. Azure / proxy). Defaults to the public OpenAI API.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # None -> api.openai.com

# RAGAS imports. Kept inside the guard so that a broken/incompatible RAGAS
# install degrades gracefully (RAGAS_AVAILABLE = False) instead of taking down
# the whole app when chat.py imports this module.
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    try:
        # Resolve the API key from the environment (chat.py sets these).
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")

        # Create evaluator LLM with model gpt-3.5-turbo (wrapped for RAGAS)
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, base_url=OPENAI_BASE_URL)
        )

        # Create evaluator_embeddings with model text-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key, base_url=OPENAI_BASE_URL)
        )

        # Define an instance for each metric to evaluate.
        # We only have (question, answer, contexts) — no ground-truth reference —
        # so we use the reference-free metrics:
        #   - Faithfulness: is the answer grounded in the retrieved contexts?
        #   - ResponseRelevancy: is the answer relevant to the question?
        # (BleuScore, RougeScore, and NonLLMContextPrecisionWithReference all
        # require a reference answer/context and are omitted here.)
        metrics = [
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ]

        # Build a single-turn evaluation sample from the inputs.
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        dataset = EvaluationDataset(samples=[sample])

        # Evaluate the response using the metrics
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        # Return the evaluation results as a {metric_name: score} dict
        scores: Dict[str, float] = {}
        df = result.to_pandas()
        for metric in metrics:
            column = metric.name
            if column in df.columns:
                value = df[column].iloc[0]
                # Guard against NaN / non-numeric results
                try:
                    scores[column] = round(float(value), 4)
                except (TypeError, ValueError):
                    scores[column] = 0.0

        return scores

    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}
