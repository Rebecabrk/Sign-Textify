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

def draw_dynamic_button(frame, text, position, padding=10, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, button_color=(0, 255, 0), text_color=(255, 255, 255)):
    """
    Draws a button on the frame that fits the text inside it.
    
    Parameters:
    - frame: The image frame on which to draw the button.
    - text: The text to display on the button.
    - position: A tuple (x, y) representing the top-left corner of the button.
    - padding: The padding around the text inside the button.
    - font: The font type for the text.
    - font_scale: The scale of the font.
    - font_thickness: The thickness of the font.
    - button_color: The color of the button in BGR format.
    - text_color: The color of the text in BGR format.
    """
    # Calculate the size of the text
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # Calculate the bottom-right corner of the button
    bottom_right = (position[0] + text_size[0] + 2 * padding, position[1] + text_size[1] + 2 * padding)
    
    # Draw the button (rectangle)
    cv2.rectangle(frame, position, bottom_right, button_color, -1)
    
    # Calculate the position for the text (centered within the button)
    text_position = (position[0] + padding, position[1] + text_size[1] + padding)
    
    # Draw the text
    cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness)
    button_bbox = (position[0], position[1], bottom_right[0], bottom_right[1])
    return frame, button_bbox
