# -*- coding: utf-8 -*-
"""
Created on Sat May 12 12:11:12 2018

@author: Hermethus
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入minst数据包
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#计算精确度
def compute_accuracy(v_xs,v_ys):
    global prediction
    #进行学习
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    #将学习结果与标签进行比较，获得一系列bool值
    #tf.argmax返回最大的那个数值所在的下标
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    #将bool值转换为0,1序列，再取平均得到精确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

#产生权重矩阵
def weight_variable(shape):
    #这个函数产生截断的正态分布，均值和标准差自己设定
    #将权重设置为截断正态分布的随机值矩阵
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#产生偏置向量
def bias_variable(shape):
    #将偏置设置为固定值
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)    

#卷积神经网络层
def conv2d(x,W):
    #strides是步长，指定卷积块在每个维度的跨度
    #strides=[1,x_movement,y_movement,1],strides[0]=strides[3]=1
    #padding指填充方式，SAME指产生的新图与原图等大
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    #strides=[1,x_movement,y_movement,1],strides[0]=strides[3]=1
    #步长为2，图片的长宽都缩减一半，维度缩减到1/4
    #ksize指kernel size卷积核大小
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

'''
卷积神经网络相当于一个滤波器，权重与偏置相当于滤波器的配置
通过对图像进行不同的滤波，获得不同的特征图

池化相当于采样，有取最大、平均、加和等手段
池化过程减小了数据量，同时损失了部分特征

经过卷积和池化，图的尺寸减小
但是，每一次卷积和池化提升数据的厚度（特征图的数量）
即，卷积和池化通过损失部分数据细节，强化特征
'''

#设置输入训练集的占位符
xs = tf.placeholder(tf.float32,[None,784])#28*28
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#重新处理图片信息
#[-1,28,28,1]
#-1表示n_samples，28*28，1表示只有一个色彩维度（灰度）（彩色则是3：RGB）
x_image = tf.reshape(xs,[-1,28,28,1])
#print(x_image.shape)#[-1,28,28,1]

#构建卷积神经网络层
#patch5*5为卷积核尺寸
#in_size=1指输入image的厚度,out_size=32输出image的厚度（特征数目）
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
#隐藏卷积层1,output:[28*28]*32长宽不变（same padding），厚度为32
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
#池化层1，output:[14*14]*32，长宽缩减到1/2，厚度不变
h_pool1 = max_pool_2x2(h_conv1)

#构建卷积神经网络层2
#patch5*5为卷积核尺寸
#in_size=32指输入image的厚度,out_size=64输出image的厚度（特征数目）
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
#隐藏卷积层2,output:[14*14]*64长宽不变（same padding），厚度为64
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
#池化层2，output:[7*7]*64，长宽缩减到1/2，厚度不变
h_pool2 = max_pool_2x2(h_conv2)

#将上述输出扁平化，再输入分类器
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

#进入分类器
#全连接层1
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#舍弃一些数据，防止过拟合，作为这一层输出
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#全连接层2，输出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#计算损失函数
#reduce_xxxx函数是对输入进行降维，通过求和或者求平均等方式
cross_entropy = -tf.reduce_sum(ys*tf.log(prediction)) #loss
#使用Adam优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(301):
    #从mnist读入数据
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1})
    if (i % 50 == 0):
        print(compute_accuracy(mnist.test.images,mnist.test.labels))














