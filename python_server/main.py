import cv2
import mediapipe as mp
import RunNN
import MediaPipeHandProcess as mphp
import Server 
from colorama import init, Fore, Style
import threading
import cv2
import json
import signal
import sys

init(autoreset=True)

with open("index_to_label.json", "r") as f:
    index_to_label = json.load(f)

start_inference = False
last_inference_label = None

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print(Fore.RED + Style.BRIGHT + "\nInterrupt received, shutting down...")
    sys.exit(0)

def listen_for_enter():
    global start_inference
    input(Fore.LIGHTMAGENTA_EX + "Press Enter to run inference on the current frame.\n")
    start_inference = True

def start_enter_listener():
    enter_thread = threading.Thread(target=listen_for_enter)
    enter_thread.start()

def main():
    global start_inference
    global last_inference_label
    # Load the ONNX model
    session = RunNN.load_model("model.onnx")

    # Mediapipe setup
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    # Start Server
    conn = Server.start_server()

    # Start a separate thread to listen for Enter key press
    start_enter_listener()

    print(Fore.LIGHTMAGENTA_EX +  "Video feed running... Press Enter to run inference on the current frame.")
    
    frame = None
    while True:
        # Receive frame from the Server
        new_frame = Server.receive_frame(conn)
        if new_frame is not None:
            frame = new_frame

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
                frame = mphp.draw_bounding_box(frame, landmarks, label)
                cv2.imshow("Live Video Feed", frame)
            else:
                print(Fore.LIGHTMAGENTA_EX + "No hand detected. Make sure your hand is visible to the camera.")
            start_enter_listener()
        else:
            if last_inference_label:
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

    conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()