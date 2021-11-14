from tracking.models.FaceDetector.opencvDNN import OpenCvFaceDetector
from tracking.models.VehicleDetector.car_classifier import OpenCVCarDetector
from tracking.target import Tracker
from tracking.box import apply_bounding_boxes
from distance import calibrate_distance
from log import getLog
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import json
import time
import cv2
import os

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def main():
    #load json
    config = load_json(os.path.join(".","settings.json"))

    logger = getLog('MAIN', config["logger_lvl"])

    ct = Tracker(
                maxItems=config['tracking']['max_items'],
                maxFramesMissing=config['tracking']['max_ttl'],
                maxPosHistory=config['tracking']['max_pos_history']
                )

    (H, W) = (None, None)

    logger.debug('Loading model')
    # filter model intialized choice goes here
    if config['model'] == "FaceDetector":
        model = OpenCvFaceDetector()
    elif config['model'] == "CarDetector":
        model = OpenCVCarDetector()

    logger.debug('Starting Video Stream')
    if type(config['video_src']) is int:
        camera = True
    else: camera = False
    if camera:
        vs = VideoStream(src=config['video_src']).start()
    else:
        vs = cv2.VideoCapture(config['video_src'])

    logger.debug('Camera warmup 2.0 seconds')
    time.sleep(2.0)

    focal_length = None
    #inital distance capture
    if config['distance']['active']:
        focal_length = calibrate_distance(vs, model, config['distance'])

    while True:
        if camera:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            if W is None or H is None:
                (H, W) = frame.shape[:2]
        else:
            _, frame = vs.read()

        #filter goes here and returns rects
        logger.debug("Grabbing rects")
        rects = model.get_rects(frame, W, H, .50)
        logger.debug(f"Rects Grabbed: {rects}")

        #update tracker
        objects = ct.update(rects)
        logger.debug(f"Objects Grabbed:")

        #draw bounding boxes
        apply_bounding_boxes(frame, objects, rects, focal_length, config["labels"], config['distance'])

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break 
    
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()