import cv2
import numpy as np

class OpenCVCarDetector:

    def __init__(self, xml=r"C:\\Users\\nickb\\code\\RoboVision\\src\\tracking\\models\\VehicleDetector\\car.xml"):
        
        self.net = cv2.CascadeClassifier(xml)

    def get_rects(self, frame, W, H, confidence):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = self.net.detectMultiScale(gray, 1.1, 1)
        rects = []
        for (x, y, w, h) in cars:
            rects.append((x,y,x+w, y+h))
        return rects
