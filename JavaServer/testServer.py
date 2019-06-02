import threading
import socket
import time


class Socket(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.PORT = 8008
        self.ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ServerSocket.bind(("", self.PORT))

    def run(self):
        while True:
            print("waiting client")
            self.ServerSocket.listen()
            client, addr = self.ServerSocket.accept()
            print(addr, "accessed")
            time.sleep(6)
            exx = client.sendall("time done \n".encode())
            print("python sending done", exx)

starting = Socket()
starting.run()

