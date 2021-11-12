import cv2
import numpy as np

class OpenCvFaceDetector:

    def __init__(self, Caffe=r"C:\\Users\\nickb\\code\\RoboVision\\src\\tracking\\models\\FaceDetector\\deploy.prototxt", model=r"C:\\Users\\nickb\\code\\RoboVision\\src\\tracking\\models\\FaceDetector\\opencv_face_detector.caffemodel"):
        
        self.net = cv2.dnn.readNetFromCaffe(Caffe, model)

    def get_rects(self, frame, W, H, confidence):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        rects = []

        for i in range(0, detections.shape[2]):
            if detections[0,0,i,2] > confidence:
                box = detections[0,0,i, 3:7] * np.array([W,H,W,H])
                rects.append(box.astype("int"))
        return rects