### 사진의 순서를 섞어주는 코드

import re
import os
import numpy as np

path = 'd:/newnew/Parksw1000/' # 원래 사진이 있는 폴더
file_list = os.listdir(path)

num_list = [i for i in range(1,1001)]
random.shuffle(num_list)

for j,i in zip(file_list,num_list): # 번호를 새로 붙여서
    rename(path+j,'d:/newnew/end/'+str(i)+'.jpg') # 새로 사진을 넣을 폴더
