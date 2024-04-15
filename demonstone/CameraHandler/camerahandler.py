import sys
sys.path.append(r"F:\python\demonstone")
from CarCommunication.processwithstop import ProcessWithStop
from CameraHandler.threadCameralanedetect import Threadcamera_lane_detect
from CameraHandler.threadCameraobjectdetect import Threadcamera_object_detect
from multiprocessing import Queue,Pipe,Process
import cv2
import base64
import numpy as np
import queue
import time
class Camera():
    def __init__(self, data_queue, queue_send):
        self.data_queue = data_queue
        self.queue_send = queue_send
        self.lane=Threadcamera_lane_detect(self.data_queue,self.queue_send)
        self.dete=Threadcamera_object_detect(self.data_queue,self.queue_send)

    def run(self): 
        # self.lane.start()
        self.dete.start()            

if __name__ == "__main__":
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
    camera_instance = Camera(data_queue, queue_send)
    
    # Start the Camera pro7cess
    camera_instance.run()

    # Wait for a certain time, or perform other operations
      # Wait for 10 seconds
    
    # Stop the Camera process
  
    # Ensure the process is terminated before exiting

    cap = cv2.VideoCapture("bfmc1.mp4")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (600, 300))
            # Convert image to bytes and then encode it as base64
            data_queue["Img1"].put(frame)
            data_queue["Img2"].put(frame)
            if not queue_send["camera"].empty():
                data=queue_send["camera"].get() 
            if not queue_send["Critical"].empty():
                data=queue_send["Critical"].get()
                print(data)
            # Convert image to bytes and then encode it as base64
    finally:
        # Stop the camera process when done
        camera_instance.stop()
        camera_instance.join()   
