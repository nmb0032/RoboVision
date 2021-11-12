""" 
Describes a target or object detected by a computer vision
algorithm. Will provide things like velocity and position
@author: Nicholas Belvin
"""
from scipy.spatial import distance as dist
from collections import OrderedDict
from dataclasses import dataclass
from queue import Queue
from time import time
import numpy as np

class Tracker:
    """
    Inspiration
    https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    """
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
        """Updates bounding boxes

        Args:
            rects (tuple): format startX, startY, endX, endY

        Returns:
            OrderedDict: Dictionary of Targets
        """
        #update list of disappeared if no objects
        if len(rects) == 0: 
            for key in self.disappeared.keys(): self.mark_disappeared(key)
            return self.objects

        inputCentroids = self.deriveCentroids(rects)

        #not tracking anything? take inputs and register all of them
        if len(self.objects) == 0:
            print("No objects registered yet")
            print(f"Input centroids: {inputCentroids}")
            for centroid in inputCentroids: self.register(centroid)
        else:
            #update
            objectIDs = list(self.objects.keys())
            objectCentroids = [val.get_pos() for val in self.objects.values()]

            #computer distance between pairs

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort() #min value in row
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for(row, col) in zip(rows, cols):
                #if seen before, skip
                if row in usedRows or col in usedCols: continue

                objectID = objectIDs[row]
                self.objects[objectID].update(inputCentroids[col], time())

                usedRows.add(row)
                usedCols.add(col)

            #rows and cols not used yet
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                    for row in unusedRows:
                        objectID = objectIDs[row]
                        self.disappeared[objectID].disappeared += 1

                        if self.disappeared[objectID].disappeared > self.maxFramesMissing:
                            self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects     
                    

            
        

    def register(self, centroid):
        print(f"Registering new target ID: {self.nextID}")
        tmp = Target(id=self.nextID, start_time=time())
        tmp.update(centroid, time())
        self.objects[self.nextID] = tmp
        self.nextID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
    
    def mark_disappeared(self, key):
        print(f"Object ID:{key} disappeared")
        self.disappeared[key].disappeared += 1

        if self.disappeared[key].disappeared > self.maxFramesMissing:
            self.deregister(key)

    def deriveCentroids(self, rects):
        inputCentroids = np.zeros((len(rects), 2), dtype=np.int)
        #derive centroid for each bounding box
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        return inputCentroids


@dataclass
class Target:
    """
    Defines a target (contour) with some extra features
    """
    MAX_POS_HIST_SIZE = 30

    id: int
    start_time: float
    disappeared: int = 0
    time_alive: float = 0.0
    pos_hist_queue: Queue = Queue(maxsize = MAX_POS_HIST_SIZE)

    def update(self, centroid, time):
        if self.pos_hist_queue.full():
            self.pos_hist_queue.get()
        self.pos_hist_queue.put(centroid)
        self.time_alive = time - self.start_time
        self.disappeared = 0

    def get_pos(self):
        if not self.pos_hist_queue.empty():
            return self.pos_hist_queue.queue[-1]
        return None

    def get_velocity(self):
        pass

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

