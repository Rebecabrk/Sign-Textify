import cv2
import sys
import numpy as np
import socket
import struct
import signal
from colorama import init, Fore

init(autoreset=True)

# Global flag to indicate if the server should stop
stop_server = False

def signal_handler(sig, frame):
    global stop_server
    print(Fore.RED + "\nInterrupt received, stopping server...")
    stop_server = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def accept_connection(host='127.0.0.1', port=5000):
    """Starts the server to receive frames."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(Fore.MAGENTA + f"Server listening on {host}:{port}")
    
    while not stop_server:
        try:
            server_socket.settimeout(1)  # Set a timeout for the accept call
            conn, addr = server_socket.accept()
            print(Fore.MAGENTA + f"Connection from {addr}")
            return conn
        except socket.timeout:
            continue  # Continue waiting for a connection
    
    print(Fore.MAGENTA + "Server stopped.")
    conn.close()
    sys.exit(1)

def receive_frame(conn):
    """Receives a single frame from the client."""
    try:
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
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
        return None