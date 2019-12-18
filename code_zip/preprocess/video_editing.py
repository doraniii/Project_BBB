
# video_editing
import os
import ssgloader
import dlib
import cv2
import numpy as np


# 동영상 편집
VIDEO_FILE_PATH = 'K:\\data\\ssgvideotest\\ssg.avi'
cap = cv2.VideoCapture(VIDEO_FILE_PATH)
if cap.isOpened() == False:
    print('Can\'t open the video')
    exit()

cap_count = 0
while(cap.isOpened()):
    ret, image = cap.read()
 
    if(int(cap.get(1)) % 10 == 0):
        #print('Saved frame number : ' + str(int(cap.get(1))))
        cv2.imwrite("K:\\data\\ssgvideotest\\images\\frame%d.jpg" % cap_count, image)
        print('Saved frame%d.jpg' % cap_count)
        cap_count += 1
