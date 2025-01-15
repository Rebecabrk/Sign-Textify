import cv2
import mediapipe as mp
import numpy as np
import RunNN
import MediaPipeHandProcess as mphp
import Server 
from colorama import init, Fore, Style
import cv2
import json
import signal
import sys
import time

def button_callback(event, x, y, flags, param):
    global start_inference, button_bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        x_min, y_min, x_max, y_max = button_bbox
        if x_min <= x <= x_max and y_min <= y <= y_max:  # Button area
            start_inference = True

def signal_handler(sig, frame):
    print(Fore.RED + Style.BRIGHT + "\nInterrupt received, shutting down...")
    conn.close()
    sys.exit(0)

def cleanup_and_exit():
    conn.close()
    cv2.destroyAllWindows()
    sys.exit(0)


def main():
    global start_inference
    global last_inference_label
    global conn
    global button_bbox

    session = RunNN.load_model("model.onnx")
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    
    cv2.namedWindow("Live Video Feed")
    cv2.setMouseCallback("Live Video Feed", button_callback)
    
    print(Fore.LIGHTMAGENTA_EX +  "Video feed running... Press 'q' to quit.")
    bbox_timestamp = None
    frame = None
    while True:
        try:
            # Receive frame from the Server
            new_frame = Server.receive_frame(conn)
            if new_frame is None:
                raise ConnectionResetError
            if new_frame is not None:
                frame = new_frame
            
            # Draw the start button
            frame, button_bbox = mphp.draw_dynamic_button(frame, "Run Inference", (10, 10))
            if frame is not None:
                # Process the frame for visualization
                landmarks_list, mediapipe_landmarks = mphp.process_frame(frame, hands)

            # Check if the user presses 'Enter' to run inference
            if start_inference:
                start_inference = False
                if landmarks_list:
                    print(Fore.LIGHTMAGENTA_EX + "Running inference...")
                    
                    # Use the first detected hand for inference
                    landmarks = landmarks_list[0]
                    label_index = RunNN.predict_label(landmarks, session)
                    label = index_to_label[str(label_index)]
                    last_inference_label = label
                    print(Fore.MAGENTA + f"Inference Result: {label}")
                    
                    # Draw the bounding box and label
                    frame_with_box = mphp.draw_bounding_box(frame, landmarks, label)
                    bbox_timestamp = time.time()
                    cv2.imshow("Live Video Feed", frame_with_box)
                else:
                    print(Fore.LIGHTMAGENTA_EX + "No hand detected. Make sure your hand is visible to the camera.")
            else:
                if last_inference_label:
                    if not (bbox_timestamp and time.time() - bbox_timestamp > 1.3):
                        frame = mphp.draw_bounding_box(frame, landmarks, last_inference_label)

                    cv2.imshow("Live Video Feed", frame)
                if mediapipe_landmarks:
                    for hand_landmarks in mediapipe_landmarks:
                        mphp.draw_hand_connections(frame, hand_landmarks)
            

            cv2.imshow("Live Video Feed", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(Fore.LIGHTMAGENTA_EX + "Exiting...")
                break
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            print(Fore.RED + "Connection interrupted. Waiting for a new connection...")
            conn.close()
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imshow("Live Video Feed", black_frame)
            cv2.waitKey(1) 
            conn = Server.accept_connection()
    cleanup_and_exit()

if __name__ == "__main__":
    start_inference = False
    last_inference_label = None
    init(autoreset=True)  # Initialize colorama

    conn = Server.accept_connection()
    signal.signal(signal.SIGINT, signal_handler) # Register signal handler for graceful shutdown

    with open("index_to_label.json", "r") as f:
        index_to_label = json.load(f)
    
    button_bbox = None

    main()