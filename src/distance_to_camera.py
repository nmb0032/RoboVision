from imutils import paths
import numpy as np
import imutils
import cv2

def find_marker(image):
        #converting image to gray scale, then blur it, then detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)

        #find contours in image and keeg the largest one
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key = cv2.contourArea())

        # Compute the bounding box of the paper region
        return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

def apply_box_and_text(frame, marker, inches):
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    cv2.putText(frame, "%.2fft" % (inches /12), (frame.shape[1] - 200, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 3)
    return frame



def main():

    # known width of object we are looking for
    # simple version of calibration of camera
    # look into Intrinsic and extrinsic parameters for camera calibration
    KNOWN_WIDTH = 12   #12 inches
    KNOWN_DISTANCE = 6 #Test note book 6 inch width

    # pass 0 for webcam or string for video
    cap = cv2.VideoCapture(0)

    #check if camera open
    if (cap.isOpened() == False):
        raise Exception("Unable to open videocapture")
    else:
        # read first frame to compute vocal point
        ret, frame = cap.read()
        marker = find_marker(frame)
        # take biggest rectangle object multiply by known distance divide by width
        focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

    while cap.isOpened():
        #capture frame
        ret, frame = cap.read()

        if ret == True:
            marker= find_marker(frame)
            inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

            frame = apply_box_and_text(frame, marker, inches)
            #display frame
            cv2.imshow('Frame', frame)

            #Q button press to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    #Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()