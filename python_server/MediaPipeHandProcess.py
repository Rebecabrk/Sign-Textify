import cv2
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def draw_bounding_box(frame, landmarks, label):
    """Draws landmarks, bounding boxes, and labels on the frame."""
    # Calculate bounding box coordinates
    x_coords = [point[0] for point in landmarks]
    y_coords = [point[1] for point in landmarks]
    x_min, x_max = int(min(x_coords) * frame.shape[1]), int(max(x_coords) * frame.shape[1])
    y_min, y_max = int(min(y_coords) * frame.shape[0]), int(max(y_coords) * frame.shape[0])

    # Draw bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display label
    cv2.putText(frame, f"Label: {label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def draw_hand_connections(frame, hand_landmarks):
    """Draws hand connections on the frame for visualization."""
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def process_frame(frame, hands):
    """Processes a single frame using Mediapipe."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        landmarks = [[(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                     for hand_landmarks in results.multi_hand_landmarks]
        return landmarks, results.multi_hand_landmarks
    return None, None
