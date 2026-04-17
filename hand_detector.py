import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Fingertip and PIP landmark indices (index, middle, ring, pinky)
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def is_hand_holding(hand_landmarks):
    """
    Detect if a single hand is holding/gripping something.

    Method: compare each fingertip's distance to palm center vs its PIP joint.
    If the tip is NOT extending past the PIP (i.e. finger is curled),
    the finger counts as gripped. 3+ curled fingers = holding.

    This is rotation-invariant and works across different hand orientations.
    """
    lm = hand_landmarks.landmark

    # Palm center: average of wrist + 4 MCP joints
    palm_x = (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5
    palm_y = (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5

    curled = 0
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        tip_dist = ((lm[tip_idx].x - palm_x) ** 2 + (lm[tip_idx].y - palm_y) ** 2) ** 0.5
        pip_dist = ((lm[pip_idx].x - palm_x) ** 2 + (lm[pip_idx].y - palm_y) ** 2) ** 0.5
        # Tip not extending clearly past PIP → finger is curled
        if tip_dist < pip_dist * 1.1:
            curled += 1

    return curled >= 3


def are_both_hands_holding(hand_landmarks_list):
    """
    Returns True only when both hands are detected AND both are holding something.
    Used by ContextAR to determine if the visitor's hands are occupied.
    """
    if len(hand_landmarks_list) < 2:
        return False
    return all(is_hand_holding(lm) for lm in hand_landmarks_list)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_list = results.multi_hand_landmarks or []
            both_holding = are_both_hands_holding(hand_list)

            # Draw landmarks
            for lm in hand_list:
                mp_draw.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Per-hand holding status labels
            for i, lm in enumerate(hand_list):
                holding = is_hand_holding(lm)
                tip = lm.landmark[FINGER_TIPS[1]]  # middle fingertip as anchor
                h, w, _ = frame.shape
                cx, cy = int(tip.x * w), int(tip.y * h) - 20
                text = "HOLDING" if holding else "FREE"
                color = (0, 0, 255) if holding else (0, 255, 0)
                cv2.putText(frame, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Main status banner
            if len(hand_list) < 2:
                label = f"HANDS: {len(hand_list)} DETECTED"
                color = (200, 200, 0)
            elif both_holding:
                label = "BOTH HANDS: OCCUPIED"
                color = (0, 0, 255)
            else:
                label = "HANDS: FREE"
                color = (0, 255, 0)

            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("ContextAR - Hand Detection", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
