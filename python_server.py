import socket
import struct
import cv2
import numpy as np
import mediapipe as mp
import signal
import sys
from colorama import init, Fore, Back, Style

# Colorama
init(autoreset=True)

# Mediapipe 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print(Fore.RED + Style.BRIGHT + "\nInterrupt received, shutting down...")
    cv2.destroyAllWindows()
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)

def process_data(conn):
    try:
        while True:
            # Read header
            header = conn.recv(5)
            if not header:
                break
            data_type, payload_size = struct.unpack('>BI', header)
            
            # Read payload
            payload = b""
            while len(payload) < payload_size:
                packet = conn.recv(payload_size - len(payload))
                if not packet:
                    break
                payload += packet

            # Initialize frames
            rgb_frame = None
            depth_frame = None
            combined_frame = None

            # Process data
            if data_type == 1:  # RGB Data
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif data_type == 2:  # Depth Data
                depth_frame = np.frombuffer(payload, dtype=np.uint16).reshape((480, 640))  
            
            # Combine data (Example: Overlay depth as a grayscale mask)
            if depth_frame is not None:
                combined_frame = cv2.applyColorMap((depth_frame / 256).astype(np.uint8), cv2.COLORMAP_JET)
                if rgb_frame is not None:
                    combined_frame = cv2.addWeighted(rgb_frame, 0.6, combined_frame, 0.4, 0)
            elif rgb_frame is not None:
                combined_frame = rgb_frame

            if combined_frame is not None:
                results = hands.process(combined_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(combined_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Show output
                cv2.imshow("Hand Tracking", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"Error: {e}")
    finally:
        conn.close()

# Server setup
def start_server(host='127.0.0.1', port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)  # Allow only one connection
    print(Fore.MAGENTA + f"Server listening on {host}:{port}")
    while True:
        try:
            conn, addr = server_socket.accept()
            print(Fore.LIGHTMAGENTA_EX + f"Connection from {addr}")
            process_data(conn)  # Directly call process_data without threading
        except KeyboardInterrupt:
            print(Fore.RED + Style.BRIGHT + "\nServer shutting down...")
            break
        finally:
            server_socket.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    start_server()
