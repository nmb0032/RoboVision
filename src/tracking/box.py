import cv2
from distance import get_distance


def apply_bounding_boxes(frame, objects, rects, focal_length, config, config_dist):
    for (objectID, target) in objects.items():
        pos = target.get_pos()
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (pos[0] - 10, pos[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if config['center_marker']:
            cv2.circle(frame, (pos[0], pos[1]), 4, (0,255,0), -1)

        if config["contour_trails"]:
            for i in range(1, len(target.pos_hist_queue.queue)):
                pos0 = target.pos_hist_queue.queue[i-1]
                pos1 = target.pos_hist_queue.queue[i]
                cv2.line(frame, pos0, pos1, (0,255,0), 2)
        

    for startX, startY, endX, endY in rects:
        if focal_length:
            dist = get_distance(focal_length, config_dist['known_width'], abs(startX - endX))
            cv2.putText(frame, f"{round(dist, 2)} CM", (30,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0),2)
