from multiprocessing import Process
from multiprocessing import Queue
from ultralytics import YOLO
import cv2
import numpy as np
class Threadcamera_object_detect(Process):
    def __init__(self,data_queue,queue_send):
        Process.__init__(self)
        self.data_queue=data_queue
        self.queue_send=queue_send
        self.model = YOLO('vip.pt')
        self.focal_length=211.75
        self.real_height=6
        self.speed=10
        self.count=0
        
    def run(self):
            while True:
                if not self.data_queue["Img2"].empty():   
                    frame=self.data_queue["Img2"].get()
                    if frame is not None:
                        results = self.model(frame,verbose=False)

    # Visualize the results on the frame
                        annotated_frame = results[0].plot()

                    # Display the annotated frame
                        cv2.imshow("YOLOv8 Inference", annotated_frame)
                        cv2.waitKey(1)

                        count1=0
                        for r in results:
                            
                            boxes = r.boxes
                            for box in boxes:
                                count1=1
                                width=box.xywh[0].tolist()[2]
                                class_name=r.names[box.cls[0].item()]  
                                distance = self.focal_length*self.real_height/width
                                print(distance)       
                                print(class_name) 
                                if class_name == "stop_sign" and distance<30 and self.count==0:
                                    data1 = {"action": "traffic_sign", "value": "stop_sign"}
                                    self.queue_send["Critical"].put(data1)
                               
                                    self.count=1
                                elif class_name == "stop_sign" and distance<30 and self.count==1:
                                    print("wait")
                                if class_name == "Crosswalk_sign" and distance<30 and self.count==0:
                                    data1 = {"action": "traffic_sign", "value": "Crosswalk_sign"}
                                    self.queue_send["Critical"].put(data1)
                                   
                                    self.count=1
                                elif class_name == "Crosswalk_sign" and distance<30 and self.count==1:
                                    print("wait")    
                                if class_name == "park" and distance<30 and self.count==0:
                                    data1 = {"action": "traffic_sign", "value": "park"}
                                    self.queue_send["Critical"].put(data1)
                                    
                                    self.count=1
                                elif class_name == "park" and distance<30 and self.count==1:
                                    print("wait")  
                                    
                                if class_name == "priority_sign" and distance<30 and self.count==0:
                                    data1 = {"action": "traffic_sign", "value": "priority_sign"}
                                    self.queue_send["Critical"].put(data1)
                                    
                                    self.count=1
                                elif class_name == "priority_sign" and distance<30 and self.count==1:
                                    print("wait")  




                        if count1==0 and self.count==1:
                            print("success")
                            self.count=0            



