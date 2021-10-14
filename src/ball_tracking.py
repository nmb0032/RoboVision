from collections import deque
from imutils.video import VideoStream
from time import sleep
import logging
import numpy as np
import json
import cv2
import imutils


def applyFilters(frame, lower, upper):

    # resize, blur, convert to hsv
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask

def get_contours(frame):
    cnts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def draw_bounding_box(frame, cnts):
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:

                cv2.circle(frame, (int(x), int(y)), int(radius),
                            (0,255,255), 2)
                cv2.circle(frame, center, 5, (0,0,255), -1)
        return frame


def load(filename):
     with open(r"./config.json") as file:
        try:
            config = json.load(file)
        except ValueError:
            logging.error("Config not valid JSON file")
            exit(-1)
        logging.info(f"Config loaded\n Content:\n {config}")
        return config

def performCapture(config):

    upperRGB = tuple(config['color']['upper'])
    lowerRGB = tuple(config['color']['lower'])
    pts      = deque(maxlen=config["buffer"])

    if config["video"] == None:
        vs = cv2.VideoCapture(0)
    else:
        vs = cv2.VideoCapture(config["video"])

    #camera warmup
    sleep(2.0)

    #loop
    while True:
        
        #grab Frame
        frame = vs.read()
        frame = frame[1] if config["video"] == None else frame

        #frame none or q pressed? end program 
        if frame is None or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #computer vision
        filter = applyFilters(frame, lowerRGB, upperRGB)
        cnts  = get_contours(filter)
        frame = draw_bounding_box(frame, cnts)

        cv2.imshow("resize, blur, hsv", frame)

    #clean up cv2 mem
    cleanup(vs)


def cleanup(vid):
    vid.release()
    cv2.destroyAllWindows()

def main():
    #logger basic
    logging.basicConfig(level=logging.INFO)
    config = load("config.json")
    performCapture(config)



    

if __name__ == "__main__":
    main()