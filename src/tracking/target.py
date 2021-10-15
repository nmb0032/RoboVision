""" 
Describes a target or object detected by a computer vision
algorithm. Will provide things like velocity and position
@author: Nicholas Belvin
"""

from dataclasses import dataclass
from queue import Queue

class Tracker:
    pass


@dataclass
class Target:
    """
    Defines a target (contour) with some extra features
    """
    MAX_POS_HIST_SIZE = 30

    id: int
    start_time: float
    time_alive: float = 0.0
    pos_hist_queue: Queue = Queue(maxsize = MAX_POS_HIST_SIZE)

    def update(self, pos, time):
        if self.pos_hist_queue.full():
            self.pos_hist_queue.get()
        self.pos_hist_queue.put(pos)
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

