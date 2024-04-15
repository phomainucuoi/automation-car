import sys
sys.path.append(r"C:\Users\Asus\OneDrive\Desktop\Daaash")
import serial
import binascii
from path_planning.localizee.CalcLidarData import CalcLidarData
import matplotlib.pyplot as plt
import math
import numpy as np
from CarCommunication.threadwithstop import ThreadWithStop
import time
from multiprocessing import Pipe
from path_planning.localizee.TImes import Timing
class locallize(ThreadWithStop):
    def __init__(self,pipeSend,pipeRecv):
        self.pipeSend=pipeSend
        self.pipeRecv=pipeRecv
        self.flag=False
        super(locallize, self).__init__()
        self.fig = plt.figure(figsize=(8,8))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_title('lidar (exit: Key E)',fontsize=18)
        plt.connect('key_press_event', lambda event: exit(1) if event.key == 'e' else None)
        self.ser=serial.Serial(port='COM10',
                    baudrate=230400,
                    timeout=5.0,
                    bytesize=8,
                    parity='N',
                    stopbits=1)
        self.tmpString = ""
        self.lines = list()
        self.angles = list()
        self.distances = list()
        self.x=None
    def run(self):
        i = 0
        while True:
            loopFlag = True
            flag2c = False
            
            if(i % 40 == 39):
                if('line' in locals()):
                    line.remove()
                line = self.ax.scatter(self.angles, self.distances, c="pink", s=5)

                self.ax.set_theta_offset(math.pi / 2)
                # plt.pause(0.1)
                self.angles.clear()
                self.distances.clear()
                i = 0
                if self.pipeRecv.poll():
                    msg=self.pipeRecv.recv()
                    self.flag=msg["value"]
            while loopFlag:
              
               
                   
                b = self.ser.read()
                tmpInt = int.from_bytes(b, 'big')
                
                if (tmpInt == 0x54):
                    self.tmpString +=  b.hex()+" "
                    flag2c = True
                    continue
                
                elif(tmpInt == 0x2c and flag2c):
                    self.tmpString += b.hex()

                    if(not len(self.tmpString[0:-5].replace(' ','')) == 90 ):
                        self.tmpString = ""
                        loopFlag = False
                        flag2c = False
                        continue
                    h=[]
                    lidarData = CalcLidarData(self.tmpString[0:-5])
                    self.angles.extend(lidarData.Angle_i)
                    self.distances.extend(lidarData.Distance_i)
                    point=[]
                    point=list(zip(lidarData.Angle_i,lidarData.Distance_i))
                    if self.flag:
                        for kk in point:
                            angle=np.degrees(kk[0])
                            if angle<=90 and angle>=0:
                                # if kk[1]>=0 and kk[1]<=1000:
                                x=kk[1]*np.sin(kk[0])
                                y=kk[1]*np.cos(kk[0])

                                if x<26 and y<43.2:
                           
                                    datax=round(14.77+x/10,3)
                                    datay= round(-13.265+y/10,3)
                                 
                                    data = {"action": "location", "value": (datax,datay)}
                                    # data =  {"action": "location", "value": (x,y)}
                                    self.pipeSend.send(data)         
                                    self.flag=False       
                         
                    del(point)  
                    self.tmpString = ""
                    loopFlag = False
                else:
                    self.tmpString += b.hex()+" "
                flag2c = False
            i +=1

 
if __name__ == "__main__":        
    pipe1,pipe2=Pipe()
    pipe3,pipe4=Pipe()
    tmm=Timing(pipe3)
    hihi=locallize(pipe1,pipe4)
    hihi.start()
    tmm.start()
    while True:
        if pipe2.poll():
            msg = pipe2.recv() 
            print(msg)
