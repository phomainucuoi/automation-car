from multiprocessing import Process
from multiprocessing import Queue,Pipe
import cv2
class Transport(Process):
    def __init__(self,pipeRecv,data_queue):
        Process.__init__(self)
        self.pipeRecv = pipeRecv
        self.data_queue = data_queue
    def run(self):
        while True:
            if self.pipeRecv.poll():
                msg = self.pipeRecv.recv() 
                if  msg["action"] == "modImg":
                    newFrame = cv2.imdecode(msg["value"], cv2.IMREAD_COLOR)
                    newFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2RGB)
                    newFrame=cv2.resize(newFrame,(600,300))
                    self.data_queue["Img1"].put(newFrame)
                    self.data_queue["Img2"].put(newFrame)
                if msg["action"]=="imu":
                   self.data_queue["yaw"].put(msg)


                
                          