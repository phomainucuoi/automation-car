# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import sys

# Append the current directory to the Python path
sys.path.append(".")

# Import required modules
from multiprocessing import Pipe
import json
from GUI.GUI_start import threadGUI_start
from CarCommunication.threadRemoteHandlerPC import threadRemoteHandlerPC
from CameraHandler.camerahandler import Camera
from path_planning.path_planing import path_plan
# Create pipes for communication
piperecvFromUI, pipesendFromUI = Pipe()
piperecvFromHandler, pipesendFromHandler = Pipe()
pipe1,pipe2=Pipe()
graph_file_path = 'Competition_track_graph.graphml'
start_node = "409"
goal_node = "408"
pathing=path_plan(piperecvFromHandler, pipesendFromUI)

# Load data from a JSON file
with open("setup/PairingData.json", "r") as file:
    data = json.load(file)

# Extract values from the loaded data
Ip = data["ip"]
Port = data["port"]
Passw = data["password"]
pathing.start()
# Create and start the GUI thread

# Create and start the remote handler thread
remoteHandlerthread = threadRemoteHandlerPC(
    piperecvFromUI, pipesendFromHandler, Ip, Port, Passw
)
remoteHandlerthread.start()

# Stop and join the GUI thread
