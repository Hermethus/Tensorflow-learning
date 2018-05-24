# -*- coding: utf-8 -*-
"""
Created on Thu May 10 20:16:34 2018

@author: Hermethus
"""

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
X = digits.data    #0~9数字图片 
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)


#定义一个隐藏层
def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    with tf.name_scope(layer_name):
        #定义权重和偏置
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
        #矩阵计算结果
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
        #dropout用于舍弃一部分数据，防止过拟合
        Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
        #输出是计算结果在激励函数之后的输出
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

#用于记录舍弃的量
keep_prob = tf.placeholder(tf.float32)
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,64])#8*8
    ys = tf.placeholder(tf.float32,[None,10])

#添加输出层
#输入是8*8的数字图片，输出是0~9的判断，隐藏层有50个神经元
l1 = add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
prediction = add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)

#loss函数,记录loss
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                                  reduction_indices=[1]))
    loss_scalar = tf.summary.scalar('loss',cross_entropy)

#训练，改变参数W&b
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)


sess = tf.Session()

#可视化文件
#在上一级目录用>tensorboard --logdir logs --host=127.0.0.1调用
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
test_writer = tf.summary.FileWriter("logs/test",sess.graph)


#训练过程
sess.run(tf.initialize_all_variables())

for i in range(1001):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.7})
    if i%100==0:
        #获得loss
        train_result = sess.run(loss_scalar,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        print(sess.run(cross_entropy,feed_dict={xs:X_train,ys:y_train,keep_prob:1}),end='\t')
        test_result = sess.run(loss_scalar,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        print(sess.run(cross_entropy,feed_dict={xs:X_test,ys:y_test,keep_prob:1}))
        #手动记录loss
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)
        
    
    

