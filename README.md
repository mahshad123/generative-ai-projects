# Generative AI Projects

Applied generative AI systems built end to end: multimodal LLM agents, retrieval-augmented generation, evaluation pipelines, and observability. Both projects were completed as part of Udacity's Generative AI program and extended into runnable, tested applications.

| Project | What it does | Stack |
|---|---|---|
| [Multimodal Content Moderation](./multimodal-content-moderation) | A FastAPI moderation service with four Gemini agents (text, image, audio, video) that screen messages for PII, tone, and quality before they reach a customer-facing LLM. Includes a Gradio app, OpenTelemetry tracing, and an LLM-as-a-judge eval suite. | Pydantic AI, Gemini 2.5, FastAPI, Gradio, Arize Phoenix, pydantic-evals, pytest |
| [NASA Mission Intelligence (RAG)](./nasa-mission-rag) | A RAG assistant over ~6 MB of Apollo 11, Apollo 13, and Challenger mission transcripts, with chunking/embedding pipeline, mission-filtered retrieval, a Streamlit chat UI, and live RAGAS scoring. | OpenAI, ChromaDB, Streamlit, RAGAS, LangChain |

## Themes across both projects

- **Structured LLM outputs**: agents return typed Pydantic models rather than free text, so downstream code can act on flags reliably.
- **Evaluation as a first-class step**: rule-based checks plus LLM-as-a-judge (moderation), and faithfulness and relevancy metrics (RAG), with repeated runs to measure non-determinism.
- **Production concerns**: API-key auth, file-type and size validation, configurable endpoints via environment variables, tracing, and graceful failure when optional dependencies are missing.

## Repository layout

```
.
├── multimodal-content-moderation/   # Gemini multimodal moderation service + evals
└── nasa-mission-rag/                # RAG over NASA mission transcripts + RAGAS evaluation
```

Each project has its own README, dependencies, and setup instructions.

## Author

**Mahshad Shariatnasab**: [github.com/mahshad123](https://github.com/mahshad123)
