import cv2
import numpy as np
import socket
import struct
from colorama import init, Fore

init(autoreset=True)

def start_server(host='127.0.0.1', port=5000):
    """Starts the server to receive frames."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(Fore.MAGENTA + f"Server listening on {host}:{port}")
    conn, addr = server_socket.accept()
    print(Fore.MAGENTA + f"Connection from {addr}")
    return conn

def receive_frame(conn):
    """Receives a single frame from the client."""
    # Read header
    header = conn.recv(5)
    if not header:
        return None
    data_type, payload_size = struct.unpack('>BI', header)
    
    # Read payload
    payload = b""
    while len(payload) < payload_size:
        packet = conn.recv(payload_size - len(payload))
        if not packet:
            return None
        payload += packet

    if data_type == 1:  # RGB Data
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        return frame
    return None
