from multiprocessing import Queue, Pipe
import numpy as np
import queue
import time
from CarCommunication.threadwithstop import ThreadWithStop
class Test(ThreadWithStop):
    def __init__(self, pipeRecv):
        super(Test, self).__init__()
        self.pipeRecv = pipeRecv
        self.count=0
        self.num1=0
        self.num2=0
    def run(self):
        while True:
            self.continous_update()
    def continous_update(self):
        if self.pipeRecv.poll():
            msg= self.pipeRecv.recv()
            if msg["action"] == "location":
                if self.count==0:
                    self.num1=msg["value"][0]
                    self.num2=msg["value"][1]
                    self.count=1
                else:
                    self.num1=0.8*self.num1+  0.2*msg["value"][0]  
                    self.num2=0.8*self.num2+  0.2*msg["value"][1]
                self.currentPos=(self.num1,self.num2)
                print(self.currentPos)   
        
