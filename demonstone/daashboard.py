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
import cv2
piperecvFromUI, pipesendFromUI = Pipe()
piperecvFromHandler, pipesendFromHandler = Pipe()
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
# camera=Camera(data_queue,queue_send)
final_decision=Finnal(queue_send,pipesendFromUI)
#################
# remoteHandlerthread.start()
# transport.start()
locall=path_plan(data_queue,queue_send)
locall.start()
# camera.run()
final_decision.start()
    ################