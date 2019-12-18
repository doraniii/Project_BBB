### SSGTEST vgg8 128 , inputsize = 128*128
# 모듈공간
import numpy as np
import collections
import csv
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import re
import tensorflow as tf
import time
import ssgloader
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#################################################

"""
import  cv2
import  os 
import  numpy  as np

path = "C:\\data\\ssg_Test1\\ssg4000\\ssg4000_shuffle\\test"

file_list = os.listdir(path)
    
for k in file_list:
    img = cv2.imread(path + '\\' + k)
    resize_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('C:\\data\\ssg_Test1\\ssg4000\\ssg4000_shuffle\\size128\\test\\' + k, resize_img)  
"""

# VGG8
import tensorflow as tf

train_image = 'C:\\data\\ssg_Test1\ssg8000\\train'
train_label = 'C:\\data\\ssg_Test1\ssg8000\\train_label.csv'
test_image = 'C:\\data\\ssg_Test1\ssg8000\\test'
test_label = 'C:\\data\\ssg_Test1\ssg8000\\\\test_label.csv'

trainX = ssgloader.image_load(train_image)
trainY = ssgloader.label_load(train_label)
testX = ssgloader.image_load(test_image)
testY = ssgloader.label_load(test_label)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

#trainX = np.vstack([trainX, train_rotate_image])
#trainY = np.vstack([trainY, trainY])

#trainX, trainY, testX, testY = ssgloader.test_set(trainX, trainY, 1000)

tf.reset_default_graph()

hidden_layer1 = 2000
hidden_layer2 = 2000

# input
x= tf.placeholder(tf.float32, [None,128,128,3]) # 
y_onehot = tf.placeholder(tf.float32, [None,4]) # onehot target값을 담는 바구니
keep_prob = tf.placeholder('float')
training = tf.placeholder(tf.bool, name='training' )
y_label = tf.argmax(y_onehot, axis = 1) # target 값 하나를 배출해서 담은 것

# conv1_1
W1_1 = tf.Variable(tf.random_normal(shape=[3,3,3,16], stddev=0.01), name='W1_1') # he 가중치 가로 세로 채널 갯수
L1_1 = tf.nn.conv2d(x,W1_1,strides=[1,1,1,1], padding='SAME')
b1_1 = tf.Variable(tf.ones([16]), name='b1_1') # 편향
L1_1 = L1_1 + b1_1
batch_z1_1 = tf.contrib.layers.batch_norm(L1_1, scale=True, is_training=training) # 배치정규화
y1_1_relu = tf.nn.leaky_relu(batch_z1_1) # relu

# conv1_2
W1_2 = tf.Variable(tf.random_normal(shape=[3,3,16,16], stddev=0.01), name='W1_2') # he 가중치
L1_2 = tf.nn.conv2d(y1_1_relu,W1_2,strides=[1,1,1,1], padding='SAME')
b1_2 = tf.Variable(tf.ones([16]), name='b1_2') # 편향
L1_2 = L1_2 + b1_2
batch_z1_2 = tf.contrib.layers.batch_norm(L1_2, scale=True, is_training=training) # 배치정규화
y1_2_relu = tf.nn.leaky_relu(batch_z1_2) # relu
L1_2 = tf.nn.max_pool(y1_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# conv2_1
W2_1 = tf.Variable(tf.random_normal(shape=[3,3,16,32], stddev=0.01), name='W2_1') # he 가중치
L2_1 = tf.nn.conv2d(L1_2,W2_1,strides=[1,1,1,1], padding='SAME')
b2_1 = tf.Variable(tf.ones([32]), name='b2_1') # 편향
L2_1 = L2_1 + b2_1
batch_z2_1 = tf.contrib.layers.batch_norm(L2_1, scale=True, is_training=training) # 배치정규화
y2_1_relu = tf.nn.leaky_relu(batch_z2_1) # relu

# conv2_2
W2_2 = tf.Variable(tf.random_normal(shape=[3,3,32,32], stddev=0.01), name='W2_2') # he 가중치
L2_2 = tf.nn.conv2d(y2_1_relu,W2_2,strides=[1,1,1,1], padding='SAME')
b2_2 = tf.Variable(tf.ones([32]), name='b2_2') # 편향
L2_2 = L2_2 + b2_2
batch_z2_2 = tf.contrib.layers.batch_norm(L2_2, scale=True, is_training=training) # 배치정규화
y2_2_relu = tf.nn.leaky_relu(batch_z2_2) # relu
L2_2 = tf.nn.max_pool(y2_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# conv3_1
W3_1 = tf.Variable(tf.random_normal(shape=[3,3,32,64], stddev=0.01), name='W3_1') # he 가중치
L3_1 = tf.nn.conv2d(L2_2,W3_1,strides=[1,1,1,1], padding='SAME')
b3_1 = tf.Variable(tf.ones([64]), name='b3_1') # 편향
L3_1 = L3_1 + b3_1
batch_z3_1 = tf.contrib.layers.batch_norm(L3_1, scale=True, is_training=training) # 배치정규화
y3_1_relu = tf.nn.leaky_relu(batch_z3_1) # relu


# conv3_2
W3_2 = tf.Variable(tf.random_normal(shape=[3,3,64,64], stddev=0.01), name='W3_2') # he 가중치
L3_2 = tf.nn.conv2d(y3_1_relu,W3_2,strides=[1,1,1,1], padding='SAME')
b3_2 = tf.Variable(tf.ones([64]), name='b3_2') # 편향
L3_2 = L3_2 + b3_2
batch_z3_2 = tf.contrib.layers.batch_norm(L3_2, scale=True, is_training=training) # 배치정규화
y3_2_relu = tf.nn.leaky_relu(batch_z3_2) # relu
L3_2 = tf.nn.max_pool(y3_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# conv4_1
W4_1 = tf.Variable(tf.random_normal(shape=[3,3,64,128], stddev=0.01), name='W4_1') # he 가중치
L4_1 = tf.nn.conv2d(L3_2,W4_1,strides=[1,1,1,1], padding='SAME')
b4_1 = tf.Variable(tf.ones([128]), name='b4_1') # 편향
L4_1 = L4_1 + b4_1
batch_z4_1 = tf.contrib.layers.batch_norm(L4_1, scale=True, is_training=training) # 배치정규화
y4_1_relu = tf.nn.leaky_relu(batch_z4_1) # relu

# conv4_2
W4_2 = tf.Variable(tf.random_normal(shape=[3,3,128,128], stddev=0.01), name='W4_2') # he 가중치
L4_2 = tf.nn.conv2d(y4_1_relu,W4_2,strides=[1,1,1,1], padding='SAME')
b4_2 = tf.Variable(tf.ones([128]), name='b4_2') # 편향
L4_2 = L4_2 + b4_2
batch_z4_2 = tf.contrib.layers.batch_norm(L4_2, scale=True, is_training=training) # 배치정규화
y4_2_relu = tf.nn.leaky_relu(batch_z4_2) # relu
L4_2 = tf.nn.max_pool(y4_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#print(y4_2_relu.shape ) #(?, 16, 16, 1024)

# FC1
W5 = tf.get_variable(name='W5', shape=[8*8*128, hidden_layer1], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 가중치
b5 = tf.Variable(tf.ones([hidden_layer1]), name='b5') # 편향
L5 = tf.reshape(L4_2,[-1,8*8*128])
y5= tf.matmul(L5,W5) + b5 # 내적
batch_z5 = tf.contrib.layers.batch_norm(y5, scale=True, is_training=training) # 배치정규화
y5_relu = tf.nn.leaky_relu(batch_z5) # relu
#y4_relu = tf.nn.relu(batch_z4) # relu
r5_drop = tf.nn.dropout(y5_relu, keep_prob)

#FC2
W6 = tf.get_variable(name='W6', shape=[hidden_layer1, hidden_layer2], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 가중치
b6 = tf.Variable(tf.ones([hidden_layer2]), name='b6') # 편향
y6 = tf.matmul(r5_drop,W6) + b6 # 내적
batch_z6 = tf.contrib.layers.batch_norm(y6, scale=True, is_training=training) # 배치정규화
y6_relu = tf.nn.leaky_relu(batch_z6) # relu
#y5_relu = tf.nn.relu(batch_z5) # relu
r6_drop = tf.nn.dropout(y6_relu, keep_prob)

# output
W7 = tf.get_variable(name='W7', shape=[hidden_layer2, 4], initializer=tf.contrib.layers.variance_scaling_initializer()) 
b7 = tf.Variable(tf.ones([4]), name='b7') 
y7= tf.matmul(r6_drop,W7) + b7
y_hat = tf.nn.softmax(y7) 

y_predict = tf.argmax(y_hat, axis=1) # 예측값을 추출
correction_prediction = tf.equal( y_predict, y_label ) # 비교
accuracy = tf.reduce_mean( tf.cast( correction_prediction, 'float' ) ) # 정확도 출력
loss = -tf.reduce_sum( y_onehot*tf.log(y_hat), axis=1 ) # 손실함수

#optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#train = optimizer.minimize(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

#모델 저장
saver = tf.train.Saver()

#실행부
init = tf.global_variables_initializer()
test_predictacc_list = []

#실행부
init = tf.global_variables_initializer()
train_acc_list = []
test_acc_list = []

with tf.Session() as sess:

    sess.run(init)
    for i in range(1,6400+1):
      
        trainX, trainY  = ssgloader.shuffle_batch(trainX, trainY)
        testX,  testY   = ssgloader.shuffle_batch(testX, testY)
       
        train_xs, train_ys  = ssgloader.next_batch(trainX, trainY,0,100)
        test_xs,  test_ys   = ssgloader.next_batch(testX, testY,0,100)
       
        sess.run(train, feed_dict={x: train_xs, y_onehot: train_ys, keep_prob:0.7, training:True})

        if  i % 64 == 0:

            print ( i/64 , 'train epoch acc:' ,sess.run(accuracy,feed_dict={x:train_xs, y_onehot: train_ys, keep_prob:1.0, training:True}))
            print ( i/64 , 'test epoch acc:' ,sess.run(accuracy,feed_dict={x:test_xs, y_onehot: test_ys, keep_prob:1.0, training:False}))
            print ( '================================================')
        
        testX,  testY   = ssgloader.shuffle_batch(testX, testY)
        test_xs,  test_ys   = ssgloader.next_batch(testX, testY,0,100)
       
    #그래프 그리는 코드
    #train_acc_list.append(sess.run(accuracy,feed_dict={x:train_xs, y_onehot: train_ys, keep_prob:1.0, training:False}))
    
    saver.save(sess, 'C:\\data\\ssg_Test1\\model9\\vgg8_128_model')
    tf.train.write_graph(sess.graph_def,".", 'C:\\data\\ssg_Test1\\model9\\vgg8_128_model.pb', as_text = False)
    tf.train.write_graph(sess.graph_def,".", 'C:\\data\\ssg_Test1\\model9\\vgg8_128_model.txt', as_text = True)
