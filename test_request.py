"""
ContextAR — Mock Client
Simulates Quest 3 sensor data and sends requests to the /ask endpoint.
Use this to demo the server without a headset.

Usage:
    python test_request.py                  # run all 5 scenarios
    python test_request.py --url http://192.168.1.42:8000
"""

import argparse
import sys

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    sys.exit(1)

SCENARIOS = [
    {
        "label": "A — Passing by (NO_RESPONSE)",
        "desc":  "Visitor glances for 2 seconds — server should stay silent.",
        "body":  {
            "question": "Tell me about this painting",
            "state": {"crowd": "low", "noise": "quiet", "gaze_duration": 2.0},
        },
    },
    {
        "label": "B — Interested visitor, low crowd (BRIEF_TEXT)",
        "desc":  "8-second gaze in a calm room — short text answer.",
        "body":  {
            "question": "Who painted this and when?",
            "state": {"crowd": "low", "noise": "noisy", "gaze_duration": 8.0},
        },
    },
    {
        "label": "C — Glancing in a crowd (GLANCE_CARD)",
        "desc":  "10-second gaze but the gallery is crowded — one-liner card.",
        "body":  {
            "question": "What is special about this painting?",
            "state": {"crowd": "crowded", "noise": "quiet", "gaze_duration": 10.0},
        },
    },
    {
        "label": "D — Deeply engaged, ideal conditions (FULL_VOICE)",
        "desc":  "20-second gaze in a calm room — full immersive answer.",
        "body":  {
            "question": "Tell me the story behind this painting.",
            "state": {"crowd": "low", "noise": "quiet", "gaze_duration": 20.0},
        },
    },
    {
        "label": "E — Engaged but crowded (BRIEF_TEXT_PROMPT)",
        "desc":  "20-second gaze but the room is packed — brief answer + quiet-spot nudge.",
        "body":  {
            "question": "What technique did the artist use?",
            "state": {"crowd": "crowded", "noise": "noisy", "gaze_duration": 20.0},
        },
    },
]

SEP = "─" * 60


def run_scenario(base_url: str, scenario: dict) -> None:
    print(f"\n{SEP}")
    print(f"  {scenario['label']}")
    print(f"  {scenario['desc']}")
    print(SEP)

    try:
        resp = requests.post(
            f"{base_url}/ask",
            json=scenario["body"],
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        print(f"  Mode    : {data.get('mode', '—')}")
        exhibit = data.get("exhibit", "")
        print(f"  Exhibit : {exhibit if exhibit else '(not identified — no image sent)'}")
        answer = data.get("answer", "")
        if answer:
            print(f"  Answer  : {answer}")
        else:
            print("  Answer  : (empty — server chose not to respond)")

    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to {base_url}. Is the server running?")
    except requests.exceptions.Timeout:
        print("  ERROR: Request timed out (30 s). Server may be overloaded.")
    except Exception as e:
        print(f"  ERROR: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ContextAR mock client")
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--scenario", type=int, choices=range(1, 6),
        help="Run only one scenario (1–5). Omit to run all."
    )
    args = parser.parse_args()

    print(f"\nContextAR Mock Client — targeting {args.url}")

    # Health check
    try:
        health = requests.get(f"{args.url}/health", timeout=5)
        print(f"Health check: {health.json()}")
    except Exception:
        print(f"WARNING: Could not reach {args.url}/health — server may be starting up.\n")

    scenarios = (
        [SCENARIOS[args.scenario - 1]] if args.scenario else SCENARIOS
    )
    for scenario in scenarios:
        run_scenario(args.url, scenario)

    print(f"\n{SEP}\nDone.\n")


if __name__ == "__main__":
    main()
