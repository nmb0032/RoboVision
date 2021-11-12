from tracking.models.FaceDetector.opencvDNN import OpenCvFaceDetector
from tracking.target import Tracker
from imutils.video import VideoStream
import logging
import numpy as np
import argparse
import imutils
import time
import cv2



def getLog(nm, loglevel="DEBUG"):
    #Creating custom logger
    logger=logging.getLogger(nm)
    #reading contents from properties file
    if loglevel=="ERROR":
        logger.setLevel(logging.ERROR)
    elif loglevel=="DEBUG":
        logger.setLevel(logging.DEBUG)
    #Creating Formatters    
    formatter=logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    #Creating Handlers
    stream_handler=logging.StreamHandler()
    #Adding Formatters to Handlers
    stream_handler.setFormatter(formatter)
    #Adding Handlers to logger
    logger.addHandler(stream_handler)
    return logger

def apply_bounding_boxes(frame, objects):
    for (objectID, target) in objects.items():
        pos = target.get_pos()
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (pos[0] - 10, pos[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (pos[0], pos[1]), 4, (0,255,0), -1)

def main():
    logger = getLog('MAIN')

    ct = Tracker()
    (H, W) = (None, None)

    logger.debug('Loading model')
    # filter model intialized choice goes here
    model = OpenCvFaceDetector()

    logger.debug('Starting Video Stream')
    vs = VideoStream(src=0).start()
    logger.debug('Camera warmup 2.0 seconds')
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        #filter goes here and returns rects
        logger.debug("Grabbing rects")
        rects = model.get_rects(frame, W, H, .50)
        logger.debug(f"Rects Grabbed: {rects}")

        #update tracker
        objects = ct.update(rects)
        logger.debug(f"Objects Grabbed: {list(objects.keys())}")

        #draw bounding boxes
        apply_bounding_boxes(frame, objects)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break 
    
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()