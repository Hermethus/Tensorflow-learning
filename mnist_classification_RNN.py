# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:33:14 2018

@author: Hermethus
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入minst数据包
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


learning_rate = 0.001
training_iters = 100000  #循环次数
batch_size = 128  #一批有128个图片

n_inputs = 28     #按列训练，每列训练28个pixel
n_steps = 28      #一个图片28列，要训练28次
n_hidden_units = 128  #隐藏层的神经元数
n_classes = 10      #输出的分类数

#定义输入
#形状[数据个数,列数（每个图片训练次数）,行数（每列像素数，每次训练一列）]
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

#RNN的处理单元是一个cell，包含输入和输出组件

#权重和偏置,使用字典存储,分为输入和输出部分
#RNN在每次迭代时使用相同的权重和偏置
weights = {
        #(28,128)
        'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
        #(128,10)
        'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
        }
biases = {
        #(128)
        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units])),
        #(10)
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes]))
        }

#定义RNN网络
def RNN(X,weights,biases):
    
    #hidden layer for input
    #X：[128batch,28steps,28inputs]
    #因为matmul是二维乘法,所以要将X转为二阶矩阵
    #X==>[128*28,28],-1表示自动计算
    X = tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,weights['in'])+biases['in']
    #X_in:[128*28,128]==>[128batch,28steps,128hidden_units]
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])
    
    '''===================================================='''
    #RNN核心组件cell,使用BasicLSTMCell
    #forget_bias为遗忘门的规模,默认设为1表示开始时不进行遗忘
    #
    #在LSTM中，每一步的state分为(m_state,c_state)两种状态
    #state_is_tuple=True,则state以元组形式存在,否则n_hidden_units扩大一倍
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    #状态赋初值为0
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    
    #dynamic_rnn会将长度不足batch_size的数据用0填充
    #time_major表示n_steps是否在输入的第一维度，本例中在第二维度
    outputs,states = tf.nn.dynamic_rnn(
            lstm_cell,X_in,initial_state = _init_state,time_major=False)
    
    '''===================================================='''
    #hidden layer for output as result
    #本例中output[-1]=states[1]
    results = tf.matmul(states[1],weights['out'])+biases['out']

    return results

pred = RNN(x,weights,biases)
'''
    计算交叉熵损失函数.交叉熵可在神经网络(机器学习)中作为损失函数
    p表示真实标记的分布，q则为训练后的模型的预测标记分布
    交叉熵损失函数可以衡量p与q的相似性。
    
    softmax是归一化函数,每个变量按e的指数归一化
    注意tf.nn.softmax_cross_entropy_with_logits(labels=, logits=,)的调用方法
    logits:神经网络最后一层的输出 labels:真实的标签
'''
cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
#计算梯度，计算每个参数的步长变化，并且计算出新的参数值
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#计算精确度
#将学习结果与标签进行比较，获得一系列bool值
#tf.argmax返回最大的那个数值所在的下标
correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
#将bool值转换为0,1序列，再取平均得到精确度
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:#设置训练的epoch
        #读入mnist数据
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        #将x输入整形为28*28的图片
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})
        if step%200==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        step+=1
