# Author: Cameron Zuziak
# Date: 6/3/20
# Description: 
# This program was developed for a personal security camera which texts the user when a person is detected.
# It uses tensorflow lite to detect people, then sends an mms with an image attachment to the end user.
#

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import glob
import smtplib
from time import sleep
import email
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import email.encoders
from datetime import datetime
from tensorflow.lite.python.interpreter import Interpreter


# method to text image via verizons free smtp to sms service
# you can use a config.py file to store and import email API keys, 
# or set up environment variables. 
def sendEmail():
    f_time = datetime.now().strftime("%A %B %d %Y @ %H:%M:%S")
    msg = MIMEMultipart()
    msg["Subject"] = f_time
    msg["From"] = "youremail@gmail.com"
    msg["To"] = "1234567890@vzwpix.com"
    text = MIMEText("WARNING! Person Detected!")
    msg.attach(text)
    fp = open('image.jpg', 'rb')
    image = MIMEImage(fp.read())
    fp.close()
    msg.attach(image)
    server = smtplib.SMTP("smtp.gmail.com:587")
    server.starttls()
    server.login("youremail@gmail.com","your gmail api key")
    server.sendmail("youremail@gmail.com", "1234567890@vzwpix.com", msg.as_string())
    server.quit()


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to stop camera
        self.stopped = False

    # Start thread that to read video frame            
    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    # method to update frame, constantly loops unless self.stopped == True
    def update(self):
        while True:
            # check if thread stopped, if so then stop video stream
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    # Return the recent frame
    def read(self):
        return self.frame

    # Mehtod to stop video stream by setting self.stopped to false 
    def stop(self):
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1280x720')
args = parser.parse_args()

# set up 
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(.5)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

# get directory
current_dir = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(current_dir,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(current_dir,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

while True:
    tmr1 = cv2.getTickCount()
    frame1 = videostream.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    # bounding box of deteced object
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0] 

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # get object name
            object_name = labels[int(classes[i])] 
            # add certainty to label
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
            # set font type and size
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
            # set label location
            label_ymin = max(ymin, labelSize[1] + 10)
            # draw box for text
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
            # draw label
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            if (object_name == 'person'):
                #videostream.record()
                cv2.imwrite('image.jpg', frame)
                sendEmail()
                time.sleep(5) 
                
            
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # framerate
    tmr2 = cv2.getTickCount()
    time1 = (tmr2-tmr1)/freq
    frame_rate_calc= 1/time1

    # Press 'x' to quit
    if cv2.waitKey(1) == ord('x'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
