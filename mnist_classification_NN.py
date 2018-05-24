# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入minst数据包
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#定义一个隐藏层
def add_layer(inputs,in_size,out_size,activation_function=None):
    #定义权重和偏置
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    #矩阵计算结果
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    #输出是计算结果在激励函数之后的输出
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#计算精确度
def compute_accuracy(v_xs,v_ys):
    global prediction
    #进行学习
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    #将学习结果与标签进行比较，获得一系列bool值
    #tf.argmax返回最大的那个数值所在的下标
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    #将bool值转换为0,1序列，再取平均得到精确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result


'''占位符，用来接收训练数据
   TensorFlow不能直接用数据集，而是用占位符构建图形，再导入数据
   [None,n]表示不规定数据数量，数据有n维'''
xs = tf.placeholder(tf.float32,[None,784])#28*28
ys = tf.placeholder(tf.float32,[None,10])

#只有一层
#输出层，in_size是xs的out_size，out_size是ts的size
#全局变量
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

#计算损失函数
#reduce_xxxx函数是对输入进行降维，通过求和或者求平均等方式
cross_entropy = -tf.reduce_sum(ys*tf.log(prediction)) #loss

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for _ in range(1001):
    #从mnist读入数据
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if (_ % 100 == 0):
        print(compute_accuracy(mnist.test.images,mnist.test.labels))


