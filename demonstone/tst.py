import sys
sys.path.append(".")
from multiprocessing import Pipe,Queue
import queue
from CameraHandler.camerahandler import Camera
from CarCommunication.threadRemoteHandlerPC import threadRemoteHandlerPC
from path_planning.path_planing import path_plan
from updatee.transport import Transport
from finaldecision.beyonce import Finnal
import json

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
locall=path_plan(data_queue,queue_send)
locall.run()