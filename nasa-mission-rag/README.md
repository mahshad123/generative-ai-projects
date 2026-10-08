# NASA Mission Intelligence: RAG with Live Evaluation

A retrieval-augmented generation (RAG) assistant that answers questions about **Apollo 11, Apollo 13, and the Challenger (STS-51L)** missions from primary-source NASA documents: air-to-ground transcripts, public affairs commentary, flight plans, and technical reports. Every answer can be scored in real time with **RAGAS** metrics, so you can see how well it is grounded in the retrieved sources.

## Architecture

```
data_text/{apollo11,apollo13,challenger}/*.txt
        │
        ▼
embedding_pipeline.py ── chunk (size/overlap, sentence-aware) ─▶ OpenAI embeddings ─▶ ChromaDB
                                                                     (text-embedding-3-small)
        ┌────────────────────────────────────────────────────────────────────────┘
        ▼
chat.py (Streamlit) ─▶ rag_client.py ── semantic search + mission filter ─▶ top-k chunks
        │                                                                   │
        │              llm_client.py ◀── system prompt + context + history ─┘
        ▼
ragas_evaluator.py ── Faithfulness · Response Relevancy ─▶ sidebar metrics
```

## Key features

- **Embedding pipeline** (`embedding_pipeline.py`): a CLI that chunks text with overlap and sentence-boundary breaks, pulls metadata (mission, data type, document category) from file paths and names, embeds in batches, and supports `skip`, `update`, and `replace` modes for incremental re-indexing, plus stats, test-query, and delete-by-source commands.
- **Retrieval** (`rag_client.py`): auto-discovers ChromaDB collections, runs semantic search with optional per-mission metadata filtering, and formats source-attributed context for the LLM.
- **Grounded generation** (`llm_client.py`): a domain system prompt that tells the model to answer from context and to say when information is missing, with multi-turn conversation history.
- **Live evaluation** (`ragas_evaluator.py`): reference-free RAGAS metrics (Faithfulness, Response Relevancy) per answer, which degrade gracefully if RAGAS is unavailable.
- **Batch evaluation** (`evaluate_batch.py`): runs a categorized test set (overview, emergency, disaster analysis, crew, technical, timeline) and writes per-question and aggregate scores to `evaluation_results.json`.
- **Streamlit UI** (`chat.py`): collection picker, model choice, top-k slider, and a metrics sidebar.

## Getting started

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
# optional: export OPENAI_BASE_URL="https://your-endpoint/v1"

# 1. Build the vector store (~6 MB of transcripts)
python embedding_pipeline.py --data-path ./data_text

# 2. Chat
streamlit run chat.py

# 3. Batch evaluation
python evaluate_batch.py
```

Useful pipeline options: `--chunk-size 500 --chunk-overlap 100 --batch-size 50 --update-mode skip|update|replace --stats-only --test-query "..."`.

## Evaluation results

Batch run over the six questions in `test_questions.json` (reference answers are in `evaluation_dataset.txt`) with `gpt-3.5-turbo`, top-3 retrieval:

| Metric | Mean |
|---|---|
| Response Relevancy | **0.85** |
| Faithfulness | 0.13 |

The answers are relevant, but faithfulness is low. The model tends to answer from its own background knowledge of these well-known missions instead of restricting itself to the retrieved transcript chunks, which are noisy OCR text and often only loosely related to high-level questions. Planned improvements:

- Hybrid retrieval (BM25 + dense) and a re-ranker so high-level questions pull summary documents, not transcript fragments
- Larger, structure-aware chunks for flight plans and reports
- A stricter, citation-required prompt and lower temperature
- Reference-based metrics (context precision/recall) using the expected answers in `evaluation_dataset.txt`

## Project structure

```
├── embedding_pipeline.py   # chunk → embed → ChromaDB (CLI)
├── rag_client.py           # collection discovery, retrieval, context formatting
├── llm_client.py           # OpenAI chat completion with context + history
├── ragas_evaluator.py      # RAGAS faithfulness / relevancy scoring
├── evaluate_batch.py       # batch evaluation over test_questions.json
├── chat.py                 # Streamlit app
├── test_questions.json     # evaluation questions by category
├── evaluation_dataset.txt  # reference answers
├── evaluation_results.json # latest batch evaluation output
└── data_text/              # NASA mission source documents (public domain)
```

Data: NASA mission transcripts and technical documents, which are public domain U.S. government works.

---
Built by **Mahshad Shariatnasab** as a project for Udacity's *Large Language Models and Text Generation* course, starting from the course's starter code.
