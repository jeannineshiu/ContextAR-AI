# ContextAR — Adaptive Museum Companion (Backend)

FastAPI backend for the XR museum companion system.  
Exhibition: **Western European Paintings, 15th–20th Century** — Metropolitan Museum of Art.

The backend receives visitor context (gaze duration, crowd density, ambient noise) from Unity on-device sensing, identifies the exhibit from a camera frame, retrieves relevant knowledge via RAG, and returns a mode decision with a length-appropriate text answer.

> **Architecture note (updated 2026-04):**  
> All on-device sensing (crowd detection, noise classification, gaze tracking) has been moved to Unity.  
> The server now acts as a pure QA + routing service — no sensors, no background threads, no audio generation.  
> Audio is handled by Meta TTS on the headset side.

---

## Project Structure

```
XRCC_project/
│
├── server.py               # FastAPI app — main entry point for Unity
├── qa_pipeline.py          # Coordinates all modules in order
├── context_router.py       # Decides response mode from gaze_duration + crowd + noise
├── rag_engine.py           # RAG: FAISS vector store + GPT-4o, mode-specific prompts
├── exhibit_recognizer.py   # GPT-4o Vision: identify painting from camera frame
├── exhibits_data.py        # Museum knowledge base (6 Met paintings, 5 sections each)
│
├── data/                   # Exhibit knowledge exported as Markdown (one file per painting)
│   ├── the_harvesters.md
│   ├── young_woman_water_pitcher.md
│   ├── aristotle_with_bust_of_homer.md
│   ├── madame_x.md
│   ├── wheat_field_cypresses.md
│   └── the_card_players.md
│
├── tests/                  # Pytest test suite
│   ├── test_server.py
│   ├── test_qa_pipeline.py
│   ├── test_router.py
│   ├── test_exhibit_recognizer.py
│   ├── test_rag_engine.py
│   ├── test_exhibits_data.py
│   └── test_hardware.py    # Real camera tests (opt-in)
│
├── Dockerfile              # Container image for the FastAPI server
├── docker-compose.yml      # One-command startup with pre-built FAISS index
├── conftest.py             # pytest config (hardware marker)
├── requirements.txt
├── test_request.py         # Mock client — simulates Quest 3 requests without a headset
├── README_SERVER.md        # Hackathon quick-start guide (3-step setup)
├── .env.example            # API key template — copy to .env and fill in
└── .gitignore
```

**Not in the repo (generated locally):**
- `faiss_index/` — built on first run from `exhibits_data.py`
- `.env` — your API keys (never commit this)

---

## Quick Start

### Option A — Docker (recommended for reviewers)

```bash
cp .env.example .env          # fill in your OPENAI_API_KEY
docker-compose up             # builds image and starts server
```

Server ready at `http://localhost:8000`. The pre-built FAISS index is mounted directly — no embedding step needed.

---

### Option B — Local Python

#### 1. Clone and create environment

```bash
git clone <repo-url>
cd XRCC_project

conda create -n contextar python=3.10
conda activate contextar
pip install -r requirements.txt
```

#### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY
```

#### 3. Build the FAISS knowledge index (first time only)

```bash
python rag_engine.py --build
```

This reads `exhibits_data.py`, calls OpenAI Embeddings, and saves the index to `faiss_index/`.  
Takes ~10 seconds. Must be re-run whenever `exhibits_data.py` is updated.

#### 4. Start the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`.  
Unity should connect to `http://<your-machine-ip>:8000`.

---

### Demo without a headset

Once the server is running, simulate all five visitor scenarios from the command line:

```bash
python test_request.py                         # run all 5 scenarios
python test_request.py --scenario 4            # FULL_VOICE only
python test_request.py --url http://192.168.1.42:8000  # remote server
```

---

## API Reference for Unity

Base URL: `http://<server-ip>:8000`

---

### `GET /health`

Health check. Unity can poll this to confirm the server is up.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /ask`

Main QA endpoint. Unity sends the visitor's question, current sensor state (measured on-device), and an optional camera frame. Returns a mode decision and a length-appropriate text answer.

**Request body:**
```json
{
  "question": "Who painted this and when?",
  "image_base64": "<base64 JPEG from camera — optional>",
  "state": {
    "crowd": "low",
    "noise": "quiet",
    "gaze_duration": 18.5
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `question` | `string` | Visitor's natural-language question |
| `image_base64` | `string \| null` | Base64 JPEG/PNG from the headset camera; omit to skip exhibit recognition |
| `state.crowd` | `"low" \| "crowded"` | Detected by Unity on-device |
| `state.noise` | `"quiet" \| "noisy"` | Detected by Unity on-device; does not affect mode (audio via earphones) |
| `state.gaze_duration` | `float` (seconds) | How long the visitor has been looking at this exhibit |

**Response:**
```json
{
  "mode": "FULL_VOICE",
  "answer": "This wheat field was painted by Vincent van Gogh in 1889...",
  "exhibit": "Wheat Field with Cypresses"
}
```

| Field | Notes |
|---|---|
| `mode` | See mode table below |
| `answer` | Text answer; empty string for `NO_RESPONSE` |
| `exhibit` | Recognised exhibit name; empty string if not identified |

**Mode values:**

| `mode` | When | What Unity should do |
|---|---|---|
| `NO_RESPONSE` | `gaze_duration < 5s` | Visitor is passing by — do not interrupt |
| `BRIEF_TEXT` | `5–15s`, low crowd | Display short text (2–3 sentences); play via Meta TTS if desired |
| `GLANCE_CARD` | `5–15s`, crowded | Show a minimal info card with one key fact |
| `FULL_VOICE` | `>15s`, low crowd | Display full answer and play immersive audio via Meta TTS |
| `BRIEF_TEXT_PROMPT` | `>15s`, crowded | Show brief text + nudge visitor toward a quieter spot |

---

## Context Routing Logic

The decision is made in `context_router._decide_mode()`. Priority rules:

```
gaze_duration < 5s                     →  NO_RESPONSE        (passing by)
5s ≤ gaze_duration < 15s, crowded      →  GLANCE_CARD        (minimal card)
5s ≤ gaze_duration < 15s, low crowd   →  BRIEF_TEXT         (short answer)
gaze_duration ≥ 15s, crowded           →  BRIEF_TEXT_PROMPT  (brief + quiet nudge)
gaze_duration ≥ 15s, low crowd        →  FULL_VOICE         (full immersive guide)
```

**Notes:**
- `noise` does **not** affect the mode — audio is delivered through earphones, so environment noise is irrelevant.
- `moderate` crowd is treated the same as `low`.
- Gaze thresholds are defined as constants in `context_router.py` (`GAZE_THRESHOLD_INTEREST = 5.0`, `GAZE_THRESHOLD_ENGAGED = 15.0`) and can be tuned without touching the logic.

---

## RAG System

### Knowledge base (`exhibits_data.py`)

Each of the six exhibits contains **five structured knowledge sections**, giving the LLM richer material to answer questions from multiple angles:

| Section | Content |
|---|---|
| `key_facts` | Dimensions, date, location, one-line identifiers |
| `visual_description` | Composition, colour palette, what the visitor sees |
| `historical_context` | Commission, era, events, collector history |
| `technique` | Medium, brushwork, perspective, methods |
| `story` | Scandals, surprising facts, legacy, auction records |

### Mode-specific prompts (`rag_engine.py`)

Rather than truncating output after the fact, each mode has a dedicated prompt that instructs the LLM to target the correct length and tone from the start:

| Mode | Prompt instruction | Target length |
|---|---|---|
| `GLANCE_CARD` | "Answer in exactly ONE sentence (max 20 words). State only the single most surprising fact." | ~20 words |
| `BRIEF_TEXT` | "Answer in 2–3 sentences (~50 words). Give the key fact and one interesting detail." | ~50 words |
| `FULL_VOICE` | "Answer in 4–6 sentences (~120–150 words). Include historical context, a story, and a closing thought." | ~150 words |
| `BRIEF_TEXT_PROMPT` | "Answer in 2–3 sentences (~50 words). End with a friendly nudge toward a quieter spot." | ~60 words |

### CLI testing

```bash
# Test a single mode
python rag_engine.py --query "Why did this painting cause a scandal?" --mode FULL_VOICE

# Rebuild index (required after editing exhibits_data.py)
python rag_engine.py --build

# Interactive mode — runs all four modes on each question
python rag_engine.py
```

---

## Demo Scenarios

These examples show how the system behaves across different visitor contexts.

### Scenario A — Visitor passing by (NO_RESPONSE)
```json
POST /ask
{
  "question": "Tell me about this painting",
  "state": { "crowd": "low", "noise": "quiet", "gaze_duration": 2.0 }
}
```
Expected: `mode = "NO_RESPONSE"`, `answer = ""`

### Scenario B — Interested visitor, low crowd (BRIEF_TEXT)
```json
POST /ask
{
  "question": "Tell me about this painting",
  "state": { "crowd": "low", "noise": "noisy", "gaze_duration": 8.0 }
}
```
Expected: `mode = "BRIEF_TEXT"`, 2–3 sentence answer

### Scenario C — Glancing visitor in a crowd (GLANCE_CARD)
```json
POST /ask
{
  "question": "Tell me about this painting",
  "state": { "crowd": "crowded", "noise": "quiet", "gaze_duration": 10.0 }
}
```
Expected: `mode = "GLANCE_CARD"`, one-sentence answer

### Scenario D — Deeply engaged, ideal conditions (FULL_VOICE)
```json
POST /ask
{
  "question": "Tell me about this painting",
  "state": { "crowd": "low", "noise": "quiet", "gaze_duration": 20.0 }
}
```
Expected: `mode = "FULL_VOICE"`, full 4–6 sentence immersive answer

### Scenario E — Deeply engaged but crowded (BRIEF_TEXT_PROMPT)
```json
POST /ask
{
  "question": "Tell me about this painting",
  "state": { "crowd": "crowded", "noise": "noisy", "gaze_duration": 20.0 }
}
```
Expected: `mode = "BRIEF_TEXT_PROMPT"`, brief answer + quiet-spot nudge

---

## Unity Integration Guide (Quest 3)

### Network setup

Quest 3 connects over Wi-Fi. The server must run on the **same network** as the headset.

```bash
# Find your machine's local IP (macOS)
ipconfig getifaddr en0
# Example output: 192.168.1.42
```

In your Unity scripts, set the base URL to `http://192.168.1.42:8000`.  
Never use `localhost` — that points inside the headset.

---

### For the Experience Layer

Unity is responsible for measuring `gaze_duration`, `crowd`, and `noise` on-device, then calling `/ask` with those values.

```csharp
// ExperienceLayerController.cs
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class ExperienceLayerController : MonoBehaviour
{
    private const string SERVER = "http://192.168.1.42:8000";

    // Call this when the visitor asks a question
    public void OnVisitorQuestion(string question, float gazeDuration,
                                  string crowdLevel, string noiseLevel)
    {
        StartCoroutine(AskServer(question, gazeDuration, crowdLevel, noiseLevel));
    }

    IEnumerator AskServer(string question, float gazeDuration,
                          string crowd, string noise)
    {
        var body = new AskRequest
        {
            question = question,
            state = new AskState
            {
                crowd         = crowd,
                noise         = noise,
                gaze_duration = gazeDuration
            }
        };

        string json = JsonUtility.ToJson(body);
        using var req = new UnityWebRequest($"{SERVER}/ask", "POST");
        req.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");

        yield return req.SendWebRequest();

        if (req.result == UnityWebRequest.Result.Success)
        {
            var resp = JsonUtility.FromJson<AskResponse>(req.downloadHandler.text);
            HandleResponse(resp);
        }
    }

    void HandleResponse(AskResponse resp)
    {
        switch (resp.mode)
        {
            case "NO_RESPONSE":
                // Do nothing — visitor is passing by
                break;

            case "GLANCE_CARD":
                ShowGlanceCard(resp.answer);
                break;

            case "BRIEF_TEXT":
                ShowBriefText(resp.answer);
                // Optionally speak via Meta TTS
                break;

            case "FULL_VOICE":
                ShowFullOverlay(resp.answer);
                // Speak via Meta TTS
                break;

            case "BRIEF_TEXT_PROMPT":
                ShowBriefText(resp.answer);   // answer already includes quiet-spot nudge
                break;
        }
    }

    void ShowGlanceCard(string text)  { /* minimal one-line card UI */ }
    void ShowBriefText(string text)   { /* short text panel */ }
    void ShowFullOverlay(string text) { /* full immersive overlay */ }
}
```

---

### Data classes (`ContextARModels.cs`)

```csharp
using System;

[Serializable]
public class AskState
{
    public string crowd;          // "low" | "crowded"
    public string noise;          // "quiet" | "noisy"
    public float  gaze_duration;  // seconds
}

[Serializable]
public class AskRequest
{
    public string   question;
    public string   image_base64;  // optional — omit to skip exhibit recognition
    public AskState state;
}

[Serializable]
public class AskResponse
{
    public string mode;     // NO_RESPONSE | BRIEF_TEXT | GLANCE_CARD | FULL_VOICE | BRIEF_TEXT_PROMPT
    public string answer;   // empty for NO_RESPONSE
    public string exhibit;  // recognised exhibit name; empty if not identified
}
```

---

## Running the Test Suite

```bash
# All unit tests (no camera needed, ~5 seconds)
python -m pytest tests/ -v

# Hardware integration tests (requires camera)
python -m pytest tests/test_hardware.py --hardware -v
```

---

## The Six Exhibits

| Painting | Artist | Year | Met Accession |
|---|---|---|---|
| The Harvesters | Pieter Bruegel the Elder | 1565 | 19.164 |
| Young Woman with a Water Pitcher | Johannes Vermeer | c. 1662 | 89.15.21 |
| Aristotle with a Bust of Homer | Rembrandt van Rijn | 1653 | 61.198 |
| Madame X (Madame Pierre Gautreau) | John Singer Sargent | 1883–84 | 16.53 |
| Wheat Field with Cypresses | Vincent van Gogh | 1889 | 49.30 |
| The Card Players | Paul Cézanne | c. 1890–95 | 61.101.1 |

To add or update exhibits, edit `exhibits_data.py` and rebuild the index:

```bash
python rag_engine.py --build
```
