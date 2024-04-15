from multiprocessing import Process
from CameraHandler.src.hoanchinhlane import Lane
import cv2
class Threadcamera_lane_detect(Process):
    def __init__(self,data_queue,queue_send):
        Process.__init__(self)
        self.data_queue=data_queue
        self.queue_send=queue_send
        self.sign_trai=False
        self.sign_phai=False
    def run(self):
        frame =None
        while True:
            if not self.data_queue["Img1"].empty():
                frame=self.data_queue["Img1"].get()
                if frame is not None:
                    lane=Lane(orig_frame=frame)
                    frame2,steer,lan_trai,lan_phai=lane.run()
                    if frame2 is not None:
                        cv2.imshow("img",frame2)
                        cv2.waitKey(1)
                    data1 = {"action": "steer", "value": steer}
                    self.queue_send["camera"].put(data1)
                    if self.sign_trai == False:
                        if lan_trai:
                            data1={"action": "lan_trai","value": True}
                            self.queue_send["Critical"].put(data1)  
                            self.sign_trai=True
                    else:
                        if lan_trai == False:
                            data1={"action": "lan_trai","value": False}
                            self.queue_send["Critical"].put(data1)
                    if self.sign_phai==False:
                        if lan_phai:
                            data1={"action": "lan_phai","value":lan_phai}  
                            self.queue_send["Critical"].put(data1)  
                            self.sign_phai=True
                    else:
                        if lan_phai == False:
                            data1={"action": "lan_phai","value": False}
                            self.queue_send["Critical"].put(data1)
   

