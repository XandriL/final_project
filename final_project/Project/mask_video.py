#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import os
import argparse
from visualize_cv import model, display_instances, class_names
import sys

from aux_functions import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mouse_pts = []


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 10, (255, 0, 0), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)


# In[7]:


# In[13]:

parser = argparse.ArgumentParser(description="SocialDistancing")
parser.add_argument(
    "--videopath", type=str, default="vid_short.mp4", help="Path to the video file"
)
args = parser.parse_args()

input_video = args.videopath

# In[14]:


stream = cv2.VideoCapture(input_video)

# get height,width and fps of input video
height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(stream.get(cv2.CAP_PROP_FPS))
# caption = '{} {:.2f}'.format(height, width)
scale_w = 1.2 / 2
scale_h = 4 / 2

SOLID_BACK_COLOR = (41, 41, 41)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_movie = cv2.VideoWriter("Pedestrian_detect.avi", fourcc, fps, (width, height))
bird_movie = cv2.VideoWriter(
    "Pedestrian_bird.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h))
)

frame_num = 0
total_pedestrians_detected = 0
total_six_feet_violations = 0
abs_six_feet_violations = 0
pedestrian_per_sec = 0

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
num_mouse_points = 0
first_frame_display = True


while True:
    frame_num += 1
    ret, frame = stream.read()
    if not ret:
        print('unable to fetch')
        break
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    if (frame_num == 1):
        # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tl, tr, p1, p2
        while True:
            image = frame
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if len(mouse_pts) == 7:
                cv2.destroyWindow("image")
                break
            first_frame_display = False
        four_points = mouse_pts

        M, Minv = get_camera_perspective(frame, four_points[0:4])

        # log

        perspective_img = cv2.warpPerspective(frame,M,(width,height))

        cv2.imshow('Perspective projection',perspective_img)

        pts = src = np.float32(np.array([four_points[4:]]))
        warped_pt = cv2.perspectiveTransform(pts, M)[0]
        d_thresh_og = np.sqrt(
            (four_points[4][0] - four_points[5][0]) ** 2
            + (four_points[4][1] - four_points[5][1]) ** 2
        )
        d_thresh = np.sqrt(
            (warped_pt[0][0] - warped_pt[1][0]) ** 2
            + (warped_pt[0][1] - warped_pt[1][1]) ** 2
        )
        bird_image = np.zeros(
            (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
        )

        bird_image[:] = SOLID_BACK_COLOR
        pedestrian_detect = frame

    print("Processing frame: ", frame_num)

    # draw polygon of ROI
    pts = np.array(
        [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
    )
    cv2.polylines(frame, [pts], True, (255, 255, 255), thickness=4)

    results = model.detect([frame], verbose=1)

    r = results[0]

    if (len(r['rois'])) > 0:
        pedestrian_detect = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        warped_pts, bird_image = plot_points_on_bird_eye_view(
            frame, r['rois'], M, scale_w, scale_h
        )
        centroids = plot_centroids(
            r['rois']
        )
        six_feet_violations= plot_lines_between_nodes(
            warped_pts, bird_image, d_thresh
        )
        plot_lines_between_nodes_on_camera(centroids,pedestrian_detect,d_thresh_og)
        # plot_violation_rectangles(pedestrian_boxes, )
        total_pedestrians_detected += len(r['rois'])
        total_six_feet_violations += six_feet_violations / fps
        abs_six_feet_violations += six_feet_violations
    # if(six_feet_violations>=1):
    #     anomaly = "YES"
    # else:
    #     anomaly = "NO"
    last_h = 75
    text = "# 6ft violations: " + str(int(total_six_feet_violations))
    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)
    text = "# No of Pedestrians: " + str(len(r['rois']))
    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

    cv2.imshow("Masked Video", pedestrian_detect)
    cv2.waitKey(1)
    output_movie.write(pedestrian_detect)
    bird_movie.write(bird_image)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
stream.release()
cv2.destroyWindow('masked_image')
