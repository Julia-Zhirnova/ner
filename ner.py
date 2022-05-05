import picamera
from picamera import PiCamera
import time
import cv2
import numpy as np
import os
from datetime import datetime

import tensorflow as tf
from utils import backbone
from NEURO.api import object_counting_api

detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

is_color_recognition_enabled = False # set it to true for enabling the color prediction for the detected objects
roi = 385 # roi line position
deviation = 2 # the constant that represents the object counting area
custom_object_name = 'Pedestrian'



# User quit method message 
print("You can press 'Q' to quit this script.")

# File for captured image
filename = './scenes/photo.png'

# Camera settimgs
cam_width = 1280
cam_height = 480

# Final image capture settings
scale_ratio = 0.5

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print ("Camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Buffer for captured image settings
img_width = int (cam_width * scale_ratio)
img_height = int (cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=(cam_width, cam_height)
camera.framerate = 20
#camera.hflip = True

t0 = datetime.now()
counter = 0
avgtime = 0
# Capture frames from the camera
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
    counter+=1
    cv2.imshow("pair", frame)    
    key = cv2.waitKey(1) & 0xFF
    object_counting_api.cumulative_object_counting_x_axis(frame, detection_graph, category_index,
                                                          is_color_recognition_enabled, roi, deviation,
                                                          custom_object_name)  # counting all the objects
    # if the `q` key was pressed, break from the loop and save last image
    if key == ord("q") :
        t1 = datetime.now()
        timediff = t1-t0
        print ("Average time between frames: " + str(avgtime))
        print ("Frames: " + str(counter) + " Time: " + str(timediff.total_seconds())+ " Average FPS: " + str(counter/timediff.total_seconds()))
        if (os.path.isdir("./scenes")==False):
            os.makedirs("./scenes")
        cv2.imwrite(filename, frame)
        exit(0)
        break
