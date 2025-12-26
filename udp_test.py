import socket
import time

ROBOT_HOST = "moonshot1.local"   # or IP
ROBOT_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Sending MOVE command...")
sock.sendto(b"0.40,0.00", (ROBOT_HOST, ROBOT_PORT))

time.sleep(2)

print("Sending STOP command...")
sock.sendto(b"0.00,0.00", (ROBOT_HOST, ROBOT_PORT))

print("Done.")
