### GUI 프로그램 생성하는 코드
## (0) PyQt5를 설치한다.
## (1) 일단 한 번 전체 코드를 돌린다.
## (2) HOMEPATH에 폴더가 생기면 MODALPATH에 모델을 넣는다.
## (3) IMAGEPATH에 test할 이미지를 넣는다. (안 넣어도 될 수도 있음)

from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap

import time
import sys
import tensorflow as tf
import cv2
import numpy as np
import os

HOMEPATH = os.getcwd()
DATAPATH = HOMEPATH + "/data"
IMAGEPATH = HOMEPATH + "/image"
LOGPATH = HOMEPATH + "/log"
MODELPATH = HOMEPATH + "/model"
plable = ["박성웅", "이정재", "최민식", "황정민"]

def vgg6(path):
    # VGG6
    tf.reset_default_graph()

    hidden_layer1 = 4096
    hidden_layer2 = 2048
    hidden_layer3 = 1024
    hidden_layer4 = 512

    # input
    x= tf.placeholder(tf.float32, [None, 128,128,3])
    keep_prob = tf.placeholder('float')
    training = tf.placeholder(tf.bool, name='training' )

    # conv1_1
    W1_1 = tf.Variable(tf.random_normal(shape=[3,3,3,256], stddev=0.01), name='W1_1')
    L1_1 = tf.nn.conv2d(x,W1_1,strides=[1,1,1,1], padding='SAME')
    b1_1 = tf.Variable(tf.ones([256]), name='b1_1')
    L1_1 = L1_1 + b1_1
    batch_z1_1 = tf.contrib.layers.batch_norm(L1_1, scale=True, is_training=training)
    y1_1_relu = tf.nn.leaky_relu(batch_z1_1) 

    # conv1_2
    W1_2 = tf.Variable(tf.random_normal(shape=[3,3,256,256], stddev=0.01), name='W1_2')
    L1_2 = tf.nn.conv2d(y1_1_relu,W1_2,strides=[1,1,1,1], padding='SAME')
    b1_2 = tf.Variable(tf.ones([256]), name='b1_2')
    L1_2 = L1_2 + b1_2
    batch_z1_2 = tf.contrib.layers.batch_norm(L1_2, scale=True, is_training=training)
    y1_2_relu = tf.nn.leaky_relu(batch_z1_2)
    L1_2 = tf.nn.max_pool(y1_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # conv2_1
    W2_1 = tf.Variable(tf.random_normal(shape=[3,3,256,128], stddev=0.01), name='W2_1')
    L2_1 = tf.nn.conv2d(L1_2,W2_1,strides=[1,1,1,1], padding='SAME')
    b2_1 = tf.Variable(tf.ones([128]), name='b2_1')
    L2_1 = L2_1 + b2_1
    batch_z2_1 = tf.contrib.layers.batch_norm(L2_1, scale=True, is_training=training)
    y2_1_relu = tf.nn.leaky_relu(batch_z2_1) 

    # conv2_2
    W2_2 = tf.Variable(tf.random_normal(shape=[3,3,128,128], stddev=0.01), name='W2_2') 
    L2_2 = tf.nn.conv2d(y2_1_relu,W2_2,strides=[1,1,1,1], padding='SAME')
    b2_2 = tf.Variable(tf.ones([128]), name='b2_2') 
    L2_2 = L2_2 + b2_2
    batch_z2_2 = tf.contrib.layers.batch_norm(L2_2, scale=True, is_training=training)
    y2_2_relu = tf.nn.leaky_relu(batch_z2_2) 
    L2_2 = tf.nn.max_pool(y2_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # conv3_1
    W3_1 = tf.Variable(tf.random_normal(shape=[3,3,128,64], stddev=0.01), name='W3_1')
    L3_1 = tf.nn.conv2d(L2_2,W3_1,strides=[1,1,1,1], padding='SAME')
    b3_1 = tf.Variable(tf.ones([64]), name='b3_1')
    L3_1 = L3_1 + b3_1
    batch_z3_1 = tf.contrib.layers.batch_norm(L3_1, scale=True, is_training=training)
    y3_1_relu = tf.nn.leaky_relu(batch_z3_1)

    # conv3_2
    W3_2 = tf.Variable(tf.random_normal(shape=[3,3,64,64], stddev=0.01), name='W3_2')
    L3_2 = tf.nn.conv2d(y3_1_relu,W3_2,strides=[1,1,1,1], padding='SAME')
    b3_2 = tf.Variable(tf.ones([64]), name='b3_2')
    L3_2 = L3_2 + b3_2
    batch_z3_2 = tf.contrib.layers.batch_norm(L3_2, scale=True, is_training=training) 
    y3_2_relu = tf.nn.leaky_relu(batch_z3_2)
    L3_2 = tf.nn.max_pool(y3_2_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # FC1
    W4 = tf.get_variable(name='W4', shape=[16*16*64, hidden_layer1], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 가중치
    b4 = tf.Variable(tf.ones([hidden_layer1]), name='b4') 
    L4 = tf.reshape(L3_2,[-1,16*16*64])
    y4= tf.matmul(L4,W4) + b4 
    batch_z4 = tf.contrib.layers.batch_norm(y4, scale=True, is_training=training)
    y4_relu = tf.nn.leaky_relu(batch_z4) 
    r4_drop = tf.nn.dropout(y4_relu, keep_prob)

    #FC2
    W5 = tf.get_variable(name='W5', shape=[hidden_layer1, hidden_layer2], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 가중치
    b5 = tf.Variable(tf.ones([hidden_layer2]), name='b5')
    y5 = tf.matmul(r4_drop,W5) + b5
    batch_z5 = tf.contrib.layers.batch_norm(y5, scale=True, is_training=training)
    y5_relu = tf.nn.leaky_relu(batch_z5)
    r5_drop = tf.nn.dropout(y5_relu, keep_prob)

    #FC3
    W6 = tf.get_variable(name='W6', shape=[hidden_layer2, hidden_layer3], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 가중치
    b6 = tf.Variable(tf.ones([hidden_layer3]), name='b6')
    y6 = tf.matmul(r5_drop, W6) + b6
    batch_z6 = tf.contrib.layers.batch_norm(y6, scale=True, is_training=training)
    y6_relu = tf.nn.leaky_relu(batch_z6)
    r6_drop = tf.nn.dropout(y6_relu, keep_prob)

    #FC4
    W7 = tf.get_variable(name='W7', shape=[hidden_layer3, hidden_layer4], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 가중치
    b7 = tf.Variable(tf.ones([hidden_layer4]), name='b7')
    y7 = tf.matmul(r6_drop,W7) + b7 
    batch_z7 = tf.contrib.layers.batch_norm(y7, scale=True, is_training=training)
    y7_relu = tf.nn.leaky_relu(batch_z7)
    r7_drop = tf.nn.dropout(y7_relu, keep_prob)

    # output
    W8 = tf.get_variable(name='W8', shape=[hidden_layer4, 4], initializer=tf.contrib.layers.variance_scaling_initializer()) 
    b8 = tf.Variable(tf.ones([4]), name='b8') 
    y8= tf.matmul(r7_drop,W8) + b8
    y_hat = tf.nn.softmax(y8) 
    y_predict = tf.argmax(y_hat, axis=1)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    test_acc_list = []

    with tf.Session() as sess: 
        saver.restore(sess, os.getcwd()+"/model"+"/model") # 막으면 실행X
        test_xs = np.array(cv2.imread(path)).reshape(1, 128, 128, 3)
        test_acc_list.append(sess.run(y_predict, feed_dict={x:test_xs, keep_prob:1.0, training:False}))        

    return test_acc_list[0][0]

def build():
    build_env(DATAPATH)
    build_env(IMAGEPATH)
    build_env(LOGPATH)
    build_env(MODELPATH)

def build_env(path):
    if not os.path.exists(path):
        try:
            print("Current Working Directory : %s"%os.getcwd())
            os.mkdir(path)
            print(path + ":: Path creation complete")
            time.sleep(1)
        except FileNotFoundError as FE:
            pass

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.InitWindow()
    def InitWindow(self):
        self.setWindowTitle('face')
        self.setGeometry(300, 300, 300, 300)
        vbox = QVBoxLayout()
        self.btn1 = QPushButton("image upload")
        self.btn1.clicked.connect(self.getImage)
        vbox.addWidget(self.btn1)
        self.labelIm=QLabel()
        self.label=QLabel()
        vbox.addWidget(self.labelIm)
        vbox.addWidget(self.label)
        self.setLayout(vbox)
        self.show()
    
    def getImage(self):
        fname, _ = QFileDialog.getOpenFileName(self, "image choice", "", "All Files(*);;Python Files (*.py)")
        self.label.setText(plable[vgg6(fname)])
        self.labelIm.setPixmap(QPixmap(fname))
        
if __name__ == "__main__":
    build()
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec())
