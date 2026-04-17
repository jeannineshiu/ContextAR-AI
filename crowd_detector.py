import cv2
from ultralytics import YOLO

# YOLOv8n is the lightest model; weights are auto-downloaded on first run
MODEL_PATH = "yolov8n.pt"

# Crowd thresholds
CROWD_THRESHOLD = 5       # >= this many people → crowded
NEAR_THRESHOLD = 3        # >= this many people → moderate

PERSON_CLASS_ID = 0       # COCO class 0 = person


def classify_crowd(count: int) -> tuple[str, tuple]:
    """Return a (label, BGR color) based on person count."""
    if count >= CROWD_THRESHOLD:
        return f"CROWDED ({count})", (0, 0, 255)
    elif count >= NEAR_THRESHOLD:
        return f"MODERATE ({count})", (0, 165, 255)
    else:
        return f"LOW ({count})", (0, 255, 0)


def run(source=0):
    """
    Run crowd detection on a camera or video file.

    Args:
        source: camera index (int) or video file path (str)
    """
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference — only detect persons (class 0) for speed
        results = model(frame, classes=[PERSON_CLASS_ID], verbose=False)[0]
        boxes = results.boxes

        person_count = len(boxes)
        label, color = classify_crowd(person_count)

        # Draw bounding boxes for each detected person
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Main status banner
        cv2.putText(frame, f"CROWD: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("ContextAR - Crowd Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_crowd_status(frame, model) -> dict:
    """
    Stateless helper for integration with other ContextAR modules.

    Args:
        frame: BGR numpy array from cv2
        model: preloaded YOLO instance

    Returns:
        {
            "count": int,
            "level": "low" | "moderate" | "crowded",
            "boxes": list of (x1, y1, x2, y2, conf)
        }
    """
    results = model(frame, classes=[PERSON_CLASS_ID], verbose=False)[0]
    boxes = results.boxes
    count = len(boxes)

    if count >= CROWD_THRESHOLD:
        level = "crowded"
    elif count >= NEAR_THRESHOLD:
        level = "moderate"
    else:
        level = "low"

    box_list = [
        (*map(int, b.xyxy[0]), float(b.conf[0]))
        for b in boxes
    ]

    return {"count": count, "level": level, "boxes": box_list}


if __name__ == "__main__":
    run(source=0)
