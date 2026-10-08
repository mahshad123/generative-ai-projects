# Multimodal Content Moderation Agent

A content moderation service that screens **text, images, audio, and video** before they reach a customer-facing LLM. Each modality has its own Gemini agent that returns a **typed, structured verdict** (PII, tone, quality, disturbing content) with a written rationale. The service runs behind an authenticated FastAPI API and is demonstrated through a customer-service training app in which a support agent chats with an AI "customer", and every message and attachment is moderated first.

## Architecture

```
 ┌──────────────────────┐   text / files    ┌──────────────────────────────┐
 │  Gradio chat UI      │ ────────────────▶ │  FastAPI moderation service  │
 │  (support agent)     │   Bearer auth     │  /api/v1/moderate_{text,     │
 └─────────┬────────────┘ ◀──────────────── │   image_file,audio_file,     │
           │            structured verdict  │   video_file}                │
           │ only if safe                   └──────────────┬───────────────┘
           ▼                                               │
 ┌──────────────────────┐                   ┌──────────────▼───────────────┐
 │ Customer agent       │                   │ Pydantic AI agents (Gemini)  │
 │ (Gemini role-play)   │                   │ text · image · audio · video │
 └──────────────────────┘                   └──────────────────────────────┘
           │                                               │
           └──────── OpenTelemetry spans ──▶ Arize Phoenix ◀┘
```

## Key features

- **Four modality-specific agents** built with Pydantic AI. Each returns a Pydantic model (`TextModerationResult`, `ImageModerationResult`, `AudioModerationResult`, `VideoModerationResult`) with boolean flags and a rationale. The audio agent also transcribes.
- **FastAPI service** with Bearer-token auth on every route, magic-byte file-type detection, and a health endpoint.
- **Gradio app** with multimodal input (upload or microphone). It blocks flagged content before it reaches the customer LLM and shows moderation feedback in a sidebar.
- **Observability**: OpenTelemetry + OpenInference spans exported to Arize Phoenix, grouped per conversation, including uploaded media.
- **Evaluation suite** (`pydantic-evals`): labelled test cases per modality, rule-based flag checks, **LLM-as-a-judge** rubrics on the rationale, and repeated runs (`EVAL_NUM_REPEATS`) to measure LLM consistency, with retry and jitter for flaky APIs.
- **Unit tests** (pytest) for schemas, agents, environment setup, and UI wiring.

## Moderation criteria

| Modality | Flags |
|---|---|
| Text | `contains_pii`, `is_unfriendly`, `is_unprofessional` |
| Image | `contains_pii` (any person), `is_disturbing`, `is_low_quality` |
| Video | `contains_pii` (faces), `is_disturbing`, `is_low_quality` |
| Audio | `transcription`, `contains_pii`, `is_unfriendly`, `is_unprofessional` |

## Getting started

Requires Python 3.12+ and a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
# install (uv recommended)
uv sync
# or: pip install -e . && pip install pytest pytest-asyncio pydantic-evals tenacity

cp env.example .env       # then fill in GEMINI_API_KEY and USER_API_KEY
```

Run everything (Phoenix UI on :6006, API on :8000, chat app on :7860):

```bash
multimodal-moderation
```

Or run the pieces separately:

```bash
multimodal-moderation-api     # FastAPI service
multimodal-moderation-chat    # Gradio app
```

Call the API directly:

```bash
curl -X POST http://localhost:8000/api/v1/moderate_text \
  -H "Authorization: Bearer $USER_API_KEY" -H "Content-Type: application/json" \
  -d '{"text": "Hi Sarah, your account number is ACM-2847639."}'

curl -X POST http://localhost:8000/api/v1/moderate_image_file \
  -H "Authorization: Bearer $USER_API_KEY" -F "file=@evals/test_data/image_with_person.jpg"
```

## Tests and evaluations

```bash
pytest tests/test_moderation_result.py   # offline schema tests
pytest                                    # full suite (calls Gemini)

python evals/text/test_cases.py           # also: evals/image, evals/audio, evals/video
```

The eval cases cover professional content, text with PII (name, address, email, phone, account number), unfriendly text, images with a person, low-quality images, audio with PII, and video with a face.

## Project structure

```
multimodal_moderation/
├── agents/          # text, image, audio, video moderation agents + customer role-play agent
├── types/           # Pydantic result schemas and model configuration
├── fastapi_app.py   # authenticated moderation API
├── gradio_app.py    # customer-service training chat UI
├── tracing.py       # OpenTelemetry → Phoenix
└── app.py           # launches Phoenix, API, and UI together
evals/               # pydantic-evals suites + labelled test media
tests/               # pytest unit/integration tests
```

---
Built by **Mahshad Shariatnasab** as the final project for Udacity's Generative AI program, starting from the course's starter code.
