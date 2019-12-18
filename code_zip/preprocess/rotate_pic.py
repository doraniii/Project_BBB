### 사진의 회전 시키는 코드 : 사진 이름이 숫자가 아닐 경우

import cv2
import random
import os

path= "C:\\Users\\Public\\Pictures\\Sample Pictures\\" # 기존의 사진이 저장된 곳
path2= "C:\\Users\\Public\\Pictures\\Sample Pictures\\new\\" # 돌린 사진을 넣을 곳

file_list = os.listdir(path)

for i in file_list:
    if '.jpg' in i:
        img = cv2.imread(path+i)
        (h,w) = img.shape[:2] # 열과 행만 뽑음
        
        center = (w/2,h/2)
        random_angle = random.choice([90,180,270])
        scale = 1.0
        
        # Perform the counter clockwise rotation holding at the center
        # random degrees
        M = cv2.getRotationMatrix2D(center,random_angle,scale) # 회전중심, 회전각도, 확대/축소값
        rotated = cv2.warpAffine(img,M,(h,w)) # 그림 rgb, 그림회전, (새세로와 새가로)
        
        # save image
        cv2.imwrite(path2 + "roate"+ i, rotated)
            
        #cv2.imshow('random rotated',rotated)
        #cv2.waitKey(0) # waits until a key is pressed
        #cv2.destroyAllWindows() # destroys the window showing image
