import sys
sys.path.append(r"F:\python\demonstone")
from path_planning.localizee.localize import locallize
from path_planning.Motion_Planning.Motion_Planning_Module import PurePursuit
from CarCommunication.threadwithstop import ThreadWithStop
import queue
from multiprocessing import Pipe
from path_planning.test_cali.test import Test
from path_planning.localizee.TImes import Timing
class path_plan(ThreadWithStop):
    def __init__(self, data_queue,queue_send):
        self.pipeSend1,self.pipeRecv1=Pipe()
        self.pipeSend2,self.pipeRecv2=Pipe()
        self.data_queue=data_queue
        self.queue_send=queue_send
        super(path_plan, self).__init__()
    def run(self):
        local=locallize(self.pipeSend1,self.pipeRecv2)
        timm=Timing(self.pipeSend2)
        graph_file_path = 'Competition_track_graph.graphml'
        start_node = "H19"
        goal_node = "H24"
        json_name="locadata.json"
        PurePursuitt = PurePursuit(graph_file_path, start_node, goal_node,self.pipeRecv1,self.data_queue,self.queue_send,json_name,)  
          
        local.start()
        PurePursuitt.start()
        timm.start()
        # PurePursuitt.start()
          
if __name__ == "__main__":
    pipe1,pipe2=Pipe()
    pipe3,pipe4=Pipe()
    queue_send = {
    "Critical": queue.Queue(),
    "camera": queue.Queue(),
    "path_planning": queue.Queue()
    }     
    data_queue={
        "Img1":queue.Queue(),
        "Img2":queue.Queue(),
        "yaw": queue.Queue()
    }
    locall=path_plan(data_queue,queue_send)
    locall.start()

        




