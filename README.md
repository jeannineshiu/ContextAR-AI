# ContextAR — Adaptive Museum Companion (Backend)

FastAPI backend for the XR museum companion system.  
Exhibition: **Western European Paintings, 15th–20th Century** — Metropolitan Museum of Art.

The backend fuses three real-time sensor streams (hand pose, crowd density, ambient noise)
into a single context state, then answers visitor questions with the appropriate response
mode: full voice guide, brief text, or XR menu overlay.

---

## Project Structure

```
XRCC_project/
│
├── server.py               # FastAPI app — main entry point for Unity
├── qa_pipeline.py          # Coordinates all four modules in order
├── context_router.py       # Decides FULL_VOICE / BRIEF_TEXT / XR_MENU
├── rag_engine.py           # RAG: FAISS vector store + GPT-4o answers
├── exhibit_recognizer.py   # GPT-4o Vision: identify painting from camera frame
├── exhibits_data.py        # Museum knowledge base (6 Met paintings)
├── hand_detector.py        # MediaPipe: detect hand grip state
├── crowd_detector.py       # YOLOv8n: count people in frame
├── noise_detector.py       # Microphone: classify ambient noise level
├── tts_engine.py           # Text-to-speech (gTTS / ElevenLabs)
│
├── tests/                  # Pytest test suite (300 unit tests)
│   ├── test_server.py
│   ├── test_qa_pipeline.py
│   ├── test_router.py
│   ├── test_exhibit_recognizer.py
│   ├── test_rag_engine.py
│   ├── test_exhibits_data.py
│   ├── test_hand_detector.py
│   ├── test_crowd_detector.py
│   ├── test_noise_detector.py
│   ├── test_tts_engine.py
│   └── test_hardware.py    # Real camera + microphone tests (opt-in)
│
├── static/
│   └── audio/              # TTS .mp3 files served to Unity via /audio/<file>
│
├── conftest.py             # pytest config (hardware marker)
├── requirements.txt
├── .env.example            # API key template — copy to .env and fill in
└── .gitignore
```

**Not in the repo (generated locally):**
- `faiss_index/` — built on first run from `exhibits_data.py`
- `yolov8n.pt` — auto-downloaded by ultralytics on first run
- `.env` — your API keys (never commit this)

---

## Quick Start

### 1. Clone and create environment

```bash
git clone <repo-url>
cd XRCC_project

conda create -n contextar python=3.10
conda activate contextar
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY
```

### 3. Build the FAISS knowledge index (first time only)

```bash
python rag_engine.py --build
```

This reads `exhibits_data.py`, calls OpenAI Embeddings, and saves the index to `faiss_index/`.
Takes ~10 seconds. Only needed once, or when `exhibits_data.py` is updated.

### 4. Start the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`.  
Unity should connect to `http://<your-machine-ip>:8000`.

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

### `GET /state`

Returns the latest fused sensor state.  
Unity can call this periodically (e.g. every 500 ms) to update the interaction layer.

**Response:**
```json
{
  "timestamp": 1700000000.0,
  "hands": {
    "detected": true,
    "both_holding": false,
    "per_hand": ["free"]
  },
  "crowd": {
    "count": 2,
    "level": "low"
  },
  "noise": {
    "db": -38.5,
    "level": "quiet",
    "centroid_hz": 620.3
  },
  "suggestion": "full_ui"
}
```

**Field guide:**

| Field | Values | Notes |
|---|---|---|
| `hands.detected` | `true / false` | Any hand visible in frame |
| `hands.both_holding` | `true / false` | Both hands gripping something |
| `hands.per_hand` | `["holding", "free"]` | One entry per detected hand |
| `crowd.level` | `"low" / "moderate" / "crowded"` | Based on person count |
| `noise.level` | `"quiet" / "moderate" / "noisy"` | Based on dBFS |
| `suggestion` | `"full_ui" / "minimal_ui" / "show_overlay"` | High-level hint for the AR layer |

---

### `POST /ask`

Main QA endpoint. Unity sends the visitor's question and current sensor state,
receives a mode decision, text answer, and optional audio URL.

**Request body:**
```json
{
  "question": "Who painted this and when?",
  "image_base64": "<base64 JPEG from camera — optional>",
  "state": {
    "crowd": "low",
    "noise": "quiet",
    "detected": true,
    "both_holding": false
  }
}
```

- `image_base64`: omit (or pass `null`) to skip exhibit recognition
- `state`: omit entirely to use safe defaults (`crowd=low`, `noise=quiet`, `both_holding=false`)

**Response:**
```json
{
  "mode": "FULL_VOICE",
  "answer": "This wheat field was painted by Vincent van Gogh in 1889...",
  "audio_url": "/audio/3f2a1c.mp3",
  "exhibit": "Wheat Field with Cypresses"
}
```

**Mode values:**

| `mode` | When | What Unity should do |
|---|---|---|
| `FULL_VOICE` | Quiet room, hands free | Display full answer + play audio from `audio_url` |
| `BRIEF_TEXT` | Noisy / moderate environment | Display short text only (`audio_url` will be `""`) |
| `XR_MENU` | Both hands occupied | Show XR menu overlay; ignore `answer` and `audio_url` |

**Audio playback:**
- If `audio_url` is non-empty, fetch `http://<server-ip>:8000<audio_url>` and play it
- Example full URL: `http://192.168.1.42:8000/audio/3f2a1c.mp3`

---

## Three Demo Scenes

These scenarios illustrate how the system behaves across different contexts.

### Scene A — Ideal conditions (FULL_VOICE)
```json
POST /ask
{
  "question": "Tell me about this painting",
  "state": { "crowd": "low", "noise": "quiet", "detected": true, "both_holding": false }
}
```
Expected: `mode = "FULL_VOICE"`, full text answer, audio URL populated.

### Scene B — Busy gallery (BRIEF_TEXT)
```json
POST /ask
{
  "question": "Tell me about this painting",
  "state": { "crowd": "crowded", "noise": "noisy", "detected": false, "both_holding": false }
}
```
Expected: `mode = "BRIEF_TEXT"`, short answer (≤ 160 chars), `audio_url = ""`.

### Scene C — Accessibility / both hands occupied (XR_MENU)
```json
POST /ask
{
  "question": "Tell me about this painting",
  "state": { "crowd": "low", "noise": "quiet", "detected": true, "both_holding": true }
}
```
Expected: `mode = "XR_MENU"`, `answer = ""`, `audio_url = ""`.

---

## Unity Integration Guide (Quest 3)

### Network setup

Quest 3 connects over Wi-Fi. The server must run on the **same network** as the headset.

1. Find your computer's local IP:
   ```bash
   # macOS
   ipconfig getifaddr en0
   # Example output: 192.168.1.42
   ```
2. In your Unity scripts, set the base URL to `http://192.168.1.42:8000`  
   (never use `localhost` — that points to inside the headset)

---

### For the Input Layer

Your job: poll `/state` every 500 ms and switch interaction modes accordingly.

| Condition | Switch to |
|---|---|
| `hands.both_holding == true` | Eye tracking / head gaze mode (hands are occupied) |
| `noise.level == "noisy"` | Suppress voice input, use gaze/head only |
| `noise.level == "moderate"` | Voice input optional |
| `noise.level == "quiet"` + `both_holding == false` | Full voice input enabled |

```csharp
// InputLayerController.cs
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class InputLayerController : MonoBehaviour
{
    private const string SERVER = "http://192.168.1.42:8000";

    void Start() => StartCoroutine(PollState());

    IEnumerator PollState()
    {
        while (true)
        {
            using var req = UnityWebRequest.Get($"{SERVER}/state");
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                var state = JsonUtility.FromJson<StateResponse>(req.downloadHandler.text);
                UpdateInputMode(state);
            }

            yield return new WaitForSeconds(0.5f);
        }
    }

    void UpdateInputMode(StateResponse state)
    {
        if (state.hands.both_holding)
        {
            // Switch to eye / head gaze interaction
        }
        else if (state.noise.level == "noisy")
        {
            // Disable voice, use gaze only
        }
        else
        {
            // Full voice input
        }
    }
}
```

---

### For the Experience Layer

Your job: call `/ask` when the visitor asks a question, then update the scene.

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
    public void OnVisitorQuestion(string question, string crowdLevel, string noiseLevel, bool bothHolding)
    {
        StartCoroutine(AskServer(question, crowdLevel, noiseLevel, bothHolding));
    }

    IEnumerator AskServer(string question, string crowd, string noise, bool bothHolding)
    {
        var body = new AskRequest
        {
            question = question,
            state = new AskState
            {
                crowd = crowd,
                noise = noise,
                detected = true,
                both_holding = bothHolding
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
            case "FULL_VOICE":
                ShowFullOverlay(resp.answer);
                if (!string.IsNullOrEmpty(resp.audio_url))
                    StartCoroutine(PlayAudio($"{SERVER}{resp.audio_url}"));
                break;

            case "BRIEF_TEXT":
                ShowBriefText(resp.answer);   // short text only, no audio
                break;

            case "XR_MENU":
                ShowXRMenu();                 // ignore answer and audio
                break;
        }
    }

    IEnumerator PlayAudio(string url)
    {
        using var req = UnityWebRequestMultimedia.GetAudioClip(url, AudioType.MPEG);
        yield return req.SendWebRequest();

        if (req.result == UnityWebRequest.Result.Success)
        {
            var clip = DownloadHandlerAudioClip.GetContent(req);
            var source = GetComponent<AudioSource>();
            source.clip = clip;
            source.Play();
        }
    }

    void ShowFullOverlay(string text) { /* update your UI panel */ }
    void ShowBriefText(string text)   { /* update your UI panel */ }
    void ShowXRMenu()                 { /* activate your XR menu GameObject */ }
}
```

---

### Data classes (paste into a separate file `ContextARModels.cs`)

```csharp
// ContextARModels.cs
using System;

[Serializable] public class HandState   { public bool detected; public bool both_holding; }
[Serializable] public class CrowdState  { public int count; public string level; }
[Serializable] public class NoiseState  { public float db; public string level; }
[Serializable] public class StateResponse
{
    public float     timestamp;
    public HandState hands;
    public CrowdState crowd;
    public NoiseState noise;
    public string    suggestion;
}

[Serializable] public class AskState   { public string crowd; public string noise; public bool detected; public bool both_holding; }
[Serializable] public class AskRequest { public string question; public string image_base64; public AskState state; }
[Serializable] public class AskResponse{ public string mode; public string answer; public string audio_url; public string exhibit; }
```

---

## Context Routing Logic

The decision is made in `context_router._decide_mode()`:

```
both_holding = true  →  XR_MENU     (highest priority, regardless of environment)
noise = noisy/moderate              →  BRIEF_TEXT
noise = quiet, hands free           →  FULL_VOICE
```

`crowd` level does **not** affect the mode — only `noise` and `both_holding` do.

---

## Running the Test Suite

```bash
# All unit tests (no camera/mic needed, ~5 seconds)
python -m pytest tests/ -v

# Hardware integration tests (requires camera + microphone)
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
