import os
import ssgloader
import dlib
import cv2
import numpy as np

# 얼굴만 가져오기

import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('K:\\data\\haarcascade_frontalface_alt.xml')
eye_casecade = cv2.CascadeClassifier('K:\\data\\haarcascade_eye.xml')

path="K:\\data\\new\\test3.jpg"

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

only_face = []
for (x,y,w,h) in faces:
    
    cropped = img[y - int(h / 5):y + h + int(h / 5), x - int(w / 5):x + w + int(w / 5)]
    face = cv2.resize(cropped, (128,128))
    only_face.append(face)
    
# VGG6
import tensorflow as tf
tf.reset_default_graph()

#############################################

hidden_layer1 = 1000
hidden_layer2 = 2000
hidden_layer3 = 1000

# input
x= tf.placeholder(tf.float32, [None,128,128,3], name='input_image')
y_onehot = tf.placeholder(tf.float32, [None,4])
keep_prob = tf.placeholder('float')
training = tf.placeholder(tf.bool, name='training' )
y_label = tf.argmax(y_onehot, axis = 1)

# conv1_1
W1_1 = tf.Variable(tf.random_normal(shape=[5,5,3,16], stddev=0.01), name='W1_1')
L1_1 = tf.nn.conv2d(x,W1_1,strides=[1,1,1,1], padding='SAME')
b1_1 = tf.Variable(tf.ones([16]), name='b1_1') # 편향
L1_1 = L1_1 + b1_1
batch_z1_1 = tf.contrib.layers.batch_norm(L1_1, scale=True, is_training=training) # 배치정규화
y1_1_relu = tf.nn.leaky_relu(batch_z1_1) # relu

# conv1_2
W1_2 = tf.Variable(tf.random_normal(shape=[5,5,16,16], stddev=0.01), name='W1_2') # he 가중치
L1_2 = tf.nn.conv2d(y1_1_relu,W1_2,strides=[1,1,1,1], padding='SAME')
b1_2 = tf.Variable(tf.ones([16]), name='b1_2') # 편향
L1_2 = L1_2 + b1_2
batch_z1_2 = tf.contrib.layers.batch_norm(L1_2, scale=True, is_training=training) # 배치정규화
y1_2_relu = tf.nn.leaky_relu(batch_z1_2) # relu
L1_2 = tf.nn.max_pool(y1_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# conv2_1
W2_1 = tf.Variable(tf.random_normal(shape=[4,4,16,32], stddev=0.01), name='W2_1') # he 가중치
L2_1 = tf.nn.conv2d(L1_2,W2_1,strides=[1,1,1,1], padding='SAME')
b2_1 = tf.Variable(tf.ones([32]), name='b2_1') # 편향
L2_1 = L2_1 + b2_1
batch_z2_1 = tf.contrib.layers.batch_norm(L2_1, scale=True, is_training=training) # 배치정규화
y2_1_relu = tf.nn.leaky_relu(batch_z2_1) # relu

# conv2_2
W2_2 = tf.Variable(tf.random_normal(shape=[4,4,32,32], stddev=0.01), name='W2_2') # he 가중치
L2_2 = tf.nn.conv2d(y2_1_relu,W2_2,strides=[1,1,1,1], padding='SAME')
b2_2 = tf.Variable(tf.ones([32]), name='b2_2') # 편향
L2_2 = L2_2 + b2_2
batch_z2_2 = tf.contrib.layers.batch_norm(L2_2, scale=True, is_training=training) # 배치정규화
y2_2_relu = tf.nn.leaky_relu(batch_z2_2) # relu
L2_2 = tf.nn.max_pool(y2_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# conv3_1
W3_1 = tf.Variable(tf.random_normal(shape=[3,3,32,16], stddev=0.01), name='W3_1') # he 가중치
L3_1 = tf.nn.conv2d(L2_2,W3_1,strides=[1,1,1,1], padding='SAME')
b3_1 = tf.Variable(tf.ones([16]), name='b3_1') # 편향
L3_1 = L3_1 + b3_1
batch_z3_1 = tf.contrib.layers.batch_norm(L3_1, scale=True, is_training=training) # 배치정규화
y3_1_relu = tf.nn.leaky_relu(batch_z3_1) # relu

# conv3_2
W3_2 = tf.Variable(tf.random_normal(shape=[3,3,16,16], stddev=0.01), name='W3_2') # he 가중치
L3_2 = tf.nn.conv2d(y3_1_relu,W3_2,strides=[1,1,1,1], padding='SAME')
b3_2 = tf.Variable(tf.ones([16]), name='b3_2') # 편향
L3_2 = L3_2 + b3_2
batch_z3_2 = tf.contrib.layers.batch_norm(L3_2, scale=True, is_training=training) # 배치정규화
y3_2_relu = tf.nn.leaky_relu(batch_z3_2) # relu
L3_2 = tf.nn.max_pool(y3_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# FC1
W4 = tf.get_variable(name='W4', shape=[16*16*16, hidden_layer1], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 가중치
b4 = tf.Variable(tf.ones([hidden_layer1]), name='b4') # 편향
L4 = tf.reshape(L3_2,[-1,16*16*16])
y4= tf.matmul(L4,W4) + b4 # 내적
batch_z4 = tf.contrib.layers.batch_norm(y4, scale=True, is_training=training) # 배치정규화
y4_relu = tf.nn.leaky_relu(batch_z4) # relu
#y4_relu = tf.nn.relu(batch_z4) # relu
r4_drop = tf.nn.dropout(y4_relu, keep_prob)

#FC2
W5 = tf.get_variable(name='W5', shape=[hidden_layer1, hidden_layer2], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 가중치
b5 = tf.Variable(tf.ones([hidden_layer2]), name='b5') # 편향
y5 = tf.matmul(r4_drop,W5) + b5 # 내적
batch_z5 = tf.contrib.layers.batch_norm(y5, scale=True, is_training=training) # 배치정규화
y5_relu = tf.nn.leaky_relu(batch_z5) # relu
#y5_relu = tf.nn.relu(batch_z5) # relu
r5_drop = tf.nn.dropout(y5_relu, keep_prob)

#FC3
W6 = tf.get_variable(name='W6', shape=[hidden_layer2, hidden_layer3], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 가중치
b6 = tf.Variable(tf.ones([hidden_layer3]), name='b6') # 편향
y6 = tf.matmul(r5_drop, W6) + b6 # 내적
batch_z6 = tf.contrib.layers.batch_norm(y6, scale=True, is_training=training) # 배치정규화
y6_relu = tf.nn.leaky_relu(batch_z6) # relu
r6_drop = tf.nn.dropout(y6_relu, keep_prob)

# output
W7 = tf.get_variable(name='W7', shape=[hidden_layer3, 4], initializer=tf.contrib.layers.variance_scaling_initializer()) 
b7 = tf.Variable(tf.ones([4]), name='b7') 
y7 = tf.matmul(r6_drop,W7) + b7
y_hat = tf.nn.softmax(y7, name='output_p') 

y_predict = tf.argmax(y_hat, axis=1, name='output_label') # 예측값을 추출

#모델 저장
saver = tf.train.Saver()

#실행부
init = tf.global_variables_initializer()


#모듈로 자른 사진 테스트

pic_list=[]
with tf.Session() as sess:
    
    sess.run(init)
    saver.restore(sess, 'C:\\data\\ssg_Test1\\model10_name2\\vgg6_128_model')
    for i in only_face:
        test_xs  = i.reshape(-1,128,128,3)
        label = sess.run( y_predict ,feed_dict={x:test_xs, keep_prob:1.0, training:False})
        predict_p = sess.run( y_hat ,feed_dict={x:test_xs, keep_prob:1.0, training:False})
        print(label)
        pic_list.append([label[0],np.max(predict_p[0])])
    
    # pic_list = [[2, 0.82108295], [1, 0.6996969], [2, 0.9998871]]
    # 결과별 라벨과 색
    namelabel = ['LeeJK','LeeJS','Mr.Kang','JungC']
    col = [(209,178,255),(225,228,0),(178,204,255),(255,144,144)]
    
    # 이미지에 사각형 + text
    idx = 0
    for (x,y,w,h) in faces:
        if pic_list[idx][1] >= 0.95:
            img = cv2.rectangle(img, (x - int(w / 5),y - int(h / 5) ), (x + w + int(w / 5),y + h + int(h / 5) ), col[pic_list[idx][0]], 2)
            img = cv2.putText(img, namelabel[pic_list[idx][0]], (x - int(w / 5),y - int(h / 5) -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col[pic_list[idx][0]], 1)
            idx += 1
        else:
            img = cv2.rectangle(img, (x - int(w / 5),y - int(h / 5) ), (x + w + int(w / 5),y + h + int(h / 5) ), (225,225,225), 2)
            
    
    cv2.imwrite('K:\\data\\new\\final_model10.jpg', img)
