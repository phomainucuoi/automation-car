from multiprocessing import Process
import time

class ProcessWithStop(Process):
    def __init__(self, *args, **kwargs):
        super(ProcessWithStop, self).__init__(*args, **kwargs)
        self._running = True

    def run(self):
        while self._running:
            print("Process is running")
            time.sleep(1)
        print("Process finished")

    def stop(self):
        self._running = False
