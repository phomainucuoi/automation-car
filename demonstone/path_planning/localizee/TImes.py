from multiprocessing import Queue, Pipe
import numpy as np
import queue
import time
from CarCommunication.threadwithstop import ThreadWithStop
class Timing(ThreadWithStop):
    def __init__(self, pipeSend):
        super(Timing, self).__init__()
        self.pipeSend = pipeSend
    def run(self):
        while True:
            time.sleep(0.1)   
            data={"value": True}
            self.pipeSend.send(data)