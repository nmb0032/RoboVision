""" 
Describes a target or object detected by a computer vision
algorithm. Will provide things like velocity and position
@author: Nicholas Belvin
"""
from scipy.spatial import distance
from collections import OrderedDict
from dataclasses import dataclass
from queue import Queue
from time import time

class Tracker:
    def __init__(self, maxItems=50, maxFramesMissing=50):
        """ Creates a Tracker instance to manage Targets

        Args:
            maxItems (int, optional): [description]. Defaults to 50.
            maxFramesMissing (int, optional): [description]. Defaults to 50.
        """
        self.nextID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxFramesMissing = maxFramesMissing
        self.maxItems = maxItems

    def update(self, rects):
        
        if len(rects) == 0:
            for objectID in list(self.dissappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxFramesMissing:
                    self.deregister(objectID)
            
            return self.objects
        
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for(i, (startX, startY, endX, endY)) in enumerate(rects):

            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [target.get_pos() for target in self.objects]

            D = distance.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID].update(inputCentroids[col])
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)


    def register(self, centroid):
        tmp = Target(id=self.nextObjectID, start_time=time())
        tmp.update(centroid)
        self.objects[self.nextObjectID] = tmp
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]


@dataclass
class Target:
    """
    Defines a target (contour) with some extra features
    """
    MAX_POS_HIST_SIZE = 30

    id: int
    start_time: float
    disappeared: 0
    time_alive: float = 0.0
    pos_hist_queue: Queue = Queue(maxsize = MAX_POS_HIST_SIZE)

    def update(self, centroid, time):
        if self.pos_hist_queue.full():
            self.pos_hist_queue.get()
        self.pos_hist_queue.put(centroid)
        self.time_alive = time - self.start_time

    def get_pos(self):
        if not self.pos_hist_queue.empty():
            return self.pos_hist_queue.queue[0]
        return None

    def get_velocity(self):
        return 0.0

    def get_accel(self):
        return 0.0

    def __str__(self):
        return ("ID: " + str(self.id) + "\n" 
            +  "Time Alive: " + str(self.time_alive) + "\n"
            +  "Current Pos: " + str(self.get_pos()) + "\n"
            +  "Velocity: " + str(self.get_velocity()) + "\n"
            +  "Acceleration: " + str(self.get_accel()) + "\n"
            +  "Past Pos: " + str(self.pos_hist_queue.queue))

def main():
    from time import time
    print("Testing Target class...")
    target = Target(id=0, start_time=time())
    print(target)
    for i in range(40):
        target.update((i,i), time())
        print(target)

if __name__ == '__main__':
    main()    

