import sys
sys.path.append(r"F:\python\boschcomputer\Computer\Dashboard")
from multiprocessing import Process
from multiprocessing import Queue,Pipe
import queue
import time
class Finnal(Process):
    def __init__(self,queue_send,pipeSend):
        Process.__init__(self)
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
    def run(self):
        data = {"action": "startEngine", "value": True}
        self.pipeSend.send(data)
        data = {"action": "speed", "value": 5}
        self.pipeSend.send(data)
        while True:
            while not self.Queue_send["Critical"].empty():
                msg=self.Queue_send["Critical"].get()
                if msg["action"]=="brake":
                    if msg["value"]==True:
                        data = {"action": "brake", "value": True}
                        self.pipeSend.send(data)
                        self.flag=False
                    elif msg["value"]==False:
                        self.flag=True
                if msg["action"]=="lan_trai":
                    if msg["value"]==True:
                        self.count=1
                    if msg["value"]==False:
                        self.count=2    
            if self.count==2:
                if self.first:
                    time.sleep(5)
                    data = {"action": "brake", "value": True}
                    self.pipeSend.send(data)
                    time.sleep(2)
                    self.do_xe()
                    self.first=False
                self.flag=False
            time.sleep(0.3)
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
            if self.flag:
                self.calculate_curve()
                        

    def calculate_curve(self):
        if  self.steer_path is not None:  
            steer=self.steer_path
            data = {"action": "steer", "value": steer}
            # print(data)
            self.pipeSend.send(data)

        if self.angular_speed is not None:
            data={"action": "speed", "value": self.angular_speed}
            self.pipeSend.send(data)    

        # if self.steer_lane is not None and self.steer_path is not None:
        #     steer=self.steer_lane*0.5+self.steer_path*0.5
        #     data = {"action": "steer", "value": steer}
        #     print(data)
        #     self.pipeSend.send(data)
        #     print(data)
        # if self.steer_lane is not None:
        #     steer =self.steer_lane
        #     data = {"action": "steer", "value": steer}
        #     print(data)
        #     self.pipeSend.send(data)
        # if self.steer_lane is None and self.steer_path is not None:  
        #     steer=self.steer_path
        #     data = {"action": "steer", "value": steer}
        #     self.pipeSend.send(data)
        #     print(data)
        self.angular_speed=None    
        self.steer_path=None
        self.steer_lane=None    
        time.sleep(0.3)
    def do_xe(self):
        text_dict = {
                "Speed": "-10",
                "Time": "6",
                "Steer": "-20",
            }
        data = {"action": "STS", "value": text_dict}
        self.pipeSend.send(data)
        time.sleep(6)
        text_dict = {
                "Speed": "-10",
                "Time": "1.7",
                "Steer": "0",
            }
        data = {"action": "STS", "value": text_dict}
        self.pipeSend.send(data)
        time.sleep(1.7)
        text_dict = {
                "Speed": "-10",
                "Time": "3.1",
                "Steer": "20",
            }
        data = {"action": "STS", "value": text_dict}
        self.pipeSend.send(data)
        time.sleep(3.1)
        text_dict = {
                "Speed": "10",
                "Time": "2",
                "Steer": "-20",
            }
        data = {"action": "STS", "value": text_dict}
        self.pipeSend.send(data)
        time.sleep(2)

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
    def action1(self):

        return 0                 
    def stop(self):
        super(Finnal, self).stop()

if __name__ == "__main__":
    queue_send = {
    "Critical": Queue(),
    "camera": Queue(),
    "path_planning": Queue()
}              
    pipe1,pipe2=Pipe()
    kkk=Finnal(queue_send,pipe1)
    kkk.start()
    # while True:
    data={"action": "brake", "value": True}
    queue_send["Critical"].put(data)
    data={"action": "brake", "value": False}
    queue_send["Critical"].put(data)
    time.sleep(0.1)
    data={"action": "steer", "value": 1233}
    data2={"action": "steer", "value": 133}
    queue_send["camera"].put(data)
    queue_send["path_planning"].put(data2)
    # data={"action": "steer", "value": 23}
    # queue_send["camera"].put(data)

    # so2=input()
    # data={"action": "steer", "value": so2}
    # queue_send["camera"].put(data)
        # if pipe2.poll():
        #     msg = pipe2.recv()
        #     print(msg)
    # kkk.stop()
    # kkk.join()

