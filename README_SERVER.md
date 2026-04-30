# ContextAR Server — Quick Start

Backend for the ContextAR XR museum companion.  
Receives visitor context from a Meta Quest 3 headset and returns adaptive answers via RAG + GPT-4o.

---

## Option A — Docker (recommended)

```bash
# 1. Copy and fill in your API key
cp .env.example .env
# open .env and replace "your_openai_api_key_here" with your real key

# 2. Start the server (builds image automatically)
docker-compose up
```

Server is ready at `http://localhost:8000`.  
The pre-built FAISS index is mounted directly — no embedding step needed.

---

## Option B — Local Python

```bash
# 1. Create environment
conda create -n contextar python=3.10
conda activate contextar
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# open .env and replace "your_openai_api_key_here" with your real key

# 3. Start the server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

---

## Demo Without a Headset

Run the mock client to simulate all five visitor scenarios:

```bash
python test_request.py
```

Example output:

```
────────────────────────────────────────────────────────────
  D — Deeply engaged, ideal conditions (FULL_VOICE)
  20-second gaze in a calm room — full immersive answer.
────────────────────────────────────────────────────────────
  Mode    : FULL_VOICE
  Exhibit : (not identified — no image sent)
  Answer  : This painting, Wheat Field with Cypresses by Vincent van Gogh,
            was painted in June 1889 while Van Gogh was a patient at the
            Saint-Paul-de-Mausole asylum...
```

Run a single scenario:

```bash
python test_request.py --scenario 4        # FULL_VOICE only
python test_request.py --url http://192.168.1.42:8000   # remote server
```

---

## API

`POST /ask`

```json
{
  "question": "Who painted this?",
  "image_base64": "<base64 JPEG — optional>",
  "state": {
    "crowd": "low",
    "noise": "quiet",
    "gaze_duration": 18.5
  }
}
```

Response:

```json
{
  "mode": "FULL_VOICE",
  "answer": "This was painted by...",
  "exhibit": "Wheat Field with Cypresses"
}
```

| `mode` | When | Length |
|---|---|---|
| `NO_RESPONSE` | gaze < 5 s | — |
| `BRIEF_TEXT` | 5–15 s, low crowd | ~50 words |
| `GLANCE_CARD` | 5–15 s, crowded | 1 sentence |
| `FULL_VOICE` | > 15 s, low crowd | ~150 words |
| `BRIEF_TEXT_PROMPT` | > 15 s, crowded | ~60 words + quiet nudge |

`GET /health` → `{ "status": "ok" }`

---

## Knowledge Base

Six paintings from the Metropolitan Museum of Art.  
Source text in `data/`. Pre-built FAISS index in `faiss_index/`.

| Painting | Artist |
|---|---|
| The Harvesters | Pieter Bruegel the Elder |
| Young Woman with a Water Pitcher | Johannes Vermeer |
| Aristotle with a Bust of Homer | Rembrandt van Rijn |
| Madame X | John Singer Sargent |
| Wheat Field with Cypresses | Vincent van Gogh |
| The Card Players | Paul Cézanne |
