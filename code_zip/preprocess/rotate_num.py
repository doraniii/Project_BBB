### 사진의 회전 시키는 코드 : 사진 이름이 숫자일 경우

import cv2
import random
import os

path= "C:\\Users\\knit\\Desktop\\ssg\\data\\ssg_4_128x128\\test128x128\\"
path2= "C:\\Users\\knit\\Desktop\\ssg\\data\\ssg_4_128x128\\test128x128_rotate\\"

file_list = os.listdir(path)

list = []
for i in range(len(file_list)):
    list.append('%d.jpg' %i)

for n in list:
    img = cv2.imread(path+n)
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    random_angle = random.choice([90,180,270])
    scale = 1.0
     
    # Perform the counter clockwise rotation holding at the center
    # random degrees
    M = cv2.getRotationMatrix2D(center, random_angle, scale)
    rotated = cv2.warpAffine(img, M, (h, w))
      
    # save image
    cv2.imwrite(path2 + "roate" +n, rotated)
        
    #cv2.imshow('random rotated',rotated)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
