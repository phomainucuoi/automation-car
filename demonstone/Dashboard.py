
import sys
sys.path.append(".")
from CarCommunication.processwithstop import ProcessWithStop
from CameraHandler.threadCameralanedetect import Threadcamera_lane_detect
from CameraHandler.threadCameraobjectdetect import Threadcamera_object_detect
from multiprocessing import Queue,Pipe,Process
from path_planning.path_planing import path_plan
from CarCommunication.threadRemoteHandlerPC import threadRemoteHandlerPC
import cv2
import base64
import numpy as np
import queue
import time
class time_delay(Process):
    def __init__(self,pipeSend,pipeRecv):
        Process.__init__(self)
        self.pipeSend = pipeSend
        self.pipeRecv=pipeRecv
    def run(self):
        time.sleep(5)
        data = {"value": True}
        self.pipeSend.send(data)
        while True:
            if self.pipeRecv.poll():
                msg=self.pipeRecv.recv()
                tim=msg["value"]
                print(123)
                time.sleep(tim)
                dat={"value": True}
                self.pipeSend.send(dat)

class Camera():
    def __init__(self, data_queue, queue_send):
        self.data_queue = data_queue
        self.queue_send = queue_send
        self.lane=Threadcamera_lane_detect(self.data_queue,self.queue_send)
        self.dete=Threadcamera_object_detect(self.data_queue,self.queue_send)

    def run(self): 
        # self.lane.start()
        self.dete.start()    
    
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
class Finnal(Process):
    def __init__(self,queue_send,pipeSend,pipeRecv,pipeSe):
        self.pipeSe=pipeSe
        Process.__init__(self)
        self.pipeRecv=pipeRecv
        self.gg=False
        self.pipeSend = pipeSend
        self.Queue_send = queue_send
        self.speed=5
        self.steer=None
        self.steer_lane=None
        self.steer_path=None
        self.angular_speed=None
        self.flag=True
        self.count=False
        self.first=True
        self.older_speed=0
        self.older_steer=12
        self.flag_speed=True
        self.flag_park=False
        self.flag_check=True
        
        self.increase_flag=False
        self.mot=0
    def run(self):
        data = {"action": "startEngine", "value": True}
        self.pipeSend.send(data)
        
        while True:

            while not self.Queue_send["Critical"].empty():
                msg=self.Queue_send["Critical"].get()
                if msg["action"]=="traffic_sign":
                    if msg["value"]=="priority_sign":
                        self.increase_flag=True
                        dat={"value":3}
                        self.pipeSe.send(dat) 
                if msg["action"]=="traffic_sign":
                    if msg["value"]=="stop_sign":
                        if self.mot==0:
                            data = {"action": "brake", "value": True}
                            print(123)
                            self.pipeSend.send(data)
                            time.sleep(3)
                            self.mot=1
                            self.older_speed=None
                            self.older_steer=None
                if msg["action"]=="traffic_sign":
                    if msg["value"]=="Crosswalk_sign":
                        self.flag_speed=False
                        dat={"value":3}
                        self.pipeSe.send(dat)    
                if msg["action"]=="traffic_sign":
                    if msg["value"]=="park":
                        data = {"action": "brake", "value": True}
                        self.pipeSend.send(data)
                        time.sleep(2)
                        self.do_xe()
                        self.first=False
                        self.flag=False
                if self.flag_park:
                    if msg["action"]=="lan_trai":
                        if msg["value"]==True:
                            self.count=1
                        if msg["value"]==False:
                            self.count=2    
            if self.count==2:
                if self.first==False:
                    self.do_xe=True
                    self.first=True
                    self.pipeSe.send(5)
                else:    
                    if self.do_xe==False:
                        data = {"action": "brake", "value": True}
                        self.pipeSend.send(data)
                        time.sleep(2)
                        self.do_xe()
                        self.first=False
                        self.flag=False
            while not self.Queue_send["path_planning"].empty():
                msg=self.Queue_send["path_planning"].get()
                if msg["action"]=="steer":
                    self.steer_path=-msg["value"]
                if msg["action"]=="speed":
                    self.angular_speed=msg["value"]

            while not self.Queue_send["camera"].empty():
                msg=self.Queue_send["camera"].get()
                if msg["action"]=="steer":
                    self.steer_lane=msg["value"]
            if self.flag_speed == False:
                if self.pipeRecv.poll():
                    msg=self.pipeRecv.recv()
                    self.flag_speed=msg["value"]
                    print("gayyyyyyyyy")    
            if self.do_xe==True:
                if self.pipeRecv.poll():
                    msg=self.pipeRecv.recv()
                    self.do_xe=not msg["value"]  
            if self.increase_flag==True:
                 if self.pipeRecv.poll():
                    msg=self.pipeRecv.recv()
                    self.increase_flag= not msg["value"]
                    print(self.increase_flag)
                    print(123)               
            if self.gg == False:
                if self.pipeRecv.poll():
                    msg=self.pipeRecv.recv()
                    self.gg=msg["value"]        
            if self.flag and self.gg:
                self.calculate_curve()    
    def calculate_curve(self):
        if self.older_speed!=5:
            data={"action": "speed", "value": 5}
            self.older_speed=5
            self.pipeSend.send(data)
    def do_xe(self):
       
        text_dict = {
                "Speed": "-10",
                "Time": "6",
                "Steer": "-20",
            }
        data = {"action": "STS", "value": text_dict}
        self.pipeSend.send(data)
        time.sleep(6+2)
        text_dict = {
                "Speed": "-10",
                "Time": "1.7",
                "Steer": "0",
            }
        data = {"action": "STS", "value": text_dict}
        self.pipeSend.send(data)
        time.sleep(1.7+2)
        text_dict = {
                "Speed": "-10",
                "Time": "3.1",
                "Steer": "20",
            }
        data = {"action": "STS", "value": text_dict}
        self.pipeSend.send(data)
        time.sleep(3.1+2)
        text_dict = {
                "Speed": "10",
                "Time": "2",
                "Steer": "-20",
            }
        data = {"action": "STS", "value": text_dict}
        self.pipeSend.send(data)
        time.sleep(2+2)                   

        # if self.older_speed!=5:
        #     if self.flag_speed==False:
        #         data={"action": "speed", "value": 5}
        #         self.older_speed=5
        #         self.pipeSend.send(data)
        #     else:
        #         if self.older_speed!=0:
        #             data={"action": "speed", "value": 0}
        #             self.older_speed=0
        #             self.pipeSend.send(data)  
        # if self.older_speed!=20:
        #     if self.increase_flag==True:
        #         data={"action": "speed", "value": 20}
        #         self.older_speed=20
        #         self.pipeSend.send(data)
        #     else:
        #         if self.older_speed!=0:
        #             data={"action": "speed", "value": 0}
        #             self.older_speed=0
        #             self.pipeSend.send(data)      
        # data={"action": "speed", "value": self.angular_speed}
        # self.pipeSend.send(data)
        # if  self.steer_path is not None:  
        #     if self.older_steer!=self.steer_path:
        #         steer=self.steer_path
        #         data = {"action": "steer", "value": steer}
        #         print(data)
        #         self.pipeSend.send(data)
        #         self.older_steer=self.steer_path
        # if self.angular_speed is not None:
        #     if self.older_speed!=self.angular_speed:
        #         data={"action": "speed", "value": self.angular_speed}
        #         if self.flag_speed== True and self.increase_flag==False:
        #             print(data)
        #             self.pipeSend.send(data)
        #             self.older_speed=self.angular_speed  
        #             self.pipeSend.send(data)
        #         if self.flag_speed==False and self.increase_flag==False:
        #             if self.older_speed!=5:
        #                 data={"action": "speed", "value": 5}
        #                 self.older_speed=5 
        #                 self.pipeSend.send(data)
        #         if self.increase_flag==True and self.flag_speed==True:
        #             if self.older_speed!=20:
        #                 data={"action": "speed", "value": 20}
        #                 self.older_speed=20
        #                 self.pipeSend.send(data)    
            #         elif self.flag_speed== True and self.increase_flag==True:
            #             data = {"action": "speed", "value": 15}
            #             print(data)
            #             self.pipeSend.send(data)    
            #         elif self.flag_speed==False:
            #             data = {"action": "speed", "value": 5}
            #             print(data)b
            #             self.pipeSend.send(data)    
 
        

        
 
        
        time.sleep(0.1)                         

# Append the current directory to the Python path

# Import required modules
from multiprocessing import Pipe
import json

# Create pipes for communication

if __name__ == '__main__':
    piperecvFromUI, pipesendFromUI = Pipe()
    piperecvFromHandler, pipesendFromHandler = Pipe()
    pipe1,pipe2=Pipe()
    pipe3,pipe4=Pipe()
    queue_send = {
        "Critical": Queue(),
        "camera": Queue(),
        "path_planning": Queue()
    }     
    data_queue={
        "Img1":Queue(),
        "Img2":Queue(),
        "yaw": Queue()
    }
    with open("setup/PairingData.json", "r") as file:
        data = json.load(file)
    Ip = data["ip"]
    Port = data["port"]
    Passw = data["password"]
    #################
    remoteHandlerthread = threadRemoteHandlerPC(
        piperecvFromUI, pipesendFromHandler, Ip, Port, Passw
    )
    transport=Transport(piperecvFromHandler,data_queue) #this function get data from handler and transfer other data to another
    remoteHandlerthread.start()
    camera=Camera(data_queue,queue_send)
    camera.run()
    transport.start()
    # locall=path_plan(data_queue,queue_send)
    # locall.start()
    final_decision=Finnal(queue_send,pipesendFromUI,pipe2,pipe3)
    final_decision.start()
    tim=time_delay(pipe1,pipe4)
    tim.start()
