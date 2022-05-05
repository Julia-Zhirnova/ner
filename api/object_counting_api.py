#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util
import picamera
from picamera import PiCamera
import time

import os
from datetime import datetime

def cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation, custom_object_name):
        total_passed_objects = 0     
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
        
        total_passed_objects = 0
        color = "waiting..."
        with detection_graph.as_default():
          with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        
        # Capture frames from the camera
        for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
                counter+=1
                cv2.imshow("pair", frame)
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, counting_result = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(boxes),
                                                                                                             np.squeeze(classes).astype(np.int32),
                                                                                                             np.squeeze(scores),
                                                                                                             category_index,
                                                                                                             x_reference = roi,
                                                                                                             deviation = deviation,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)
                               
                # when the object passed over line and counted, make the color of ROI line green
                if counter == 1:
                  cv2.line(input_frame, (roi, 0), (roi, height), (0, 0xFF, 0), 5)
                else:
                  cv2.line(input_frame, (roi, 0), (roi, height), (0, 0, 0xFF), 5)

                total_passed_objects = total_passed_objects + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected ' + custom_object_name + ': ' + str(total_passed_objects) + ' fps ' + str(fps),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi-10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )
                cv2.imshow('Videoout', input_frame)
                key = cv2.waitKey(1) & 0xFF
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
     
           
