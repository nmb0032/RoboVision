import cv2
import imutils

def get_focal_length(measured_distance, real_width, width_in_pixels):
    return (width_in_pixels * measured_distance) / real_width

def get_distance(focal_length, real_width, pixel_width):
    return (real_width * focal_length) / pixel_width

def calibrate_distance(cap, model, config):

    while True:
        frame = cap.read()
        frame = imutils.resize(frame, width=400)
        (H, W) = frame.shape[:2]

        rects = model.get_rects(frame, W, H, .50)

        #draw bounding boxes
        w = apply_simple_box(frame, rects)

        cv2.imshow("Calibration Frame, press q when object set", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break 
    
    cv2.destroyAllWindows()
    return get_focal_length(config['known_distance'], config['known_width'], w)



def apply_simple_box(frame, rects):
    for startX, startY, endX, endY in rects:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0),2)
        return abs(endX - startX)