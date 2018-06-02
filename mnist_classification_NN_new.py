# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:24:45 2018

@author: Hermethus
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#相关常数
INPUT_NODE = 784
OUTPUT_NODE = 10

#神经网络参数
LAYER1_NODE = 128
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8    #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率衰减率
REGULARIZATION_RATE = 0.0001#模型复杂度正则化项在损失函数中的系数
                            #用于抑制过拟合
TRAINING_STEPS = 3001

MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率，滑动平均模型类的参数
                            #用于控制模型更新速度
                            #越大模型越趋于稳定，随训练过程增大

'''
    定义了一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的
    前向传播结果。在这里使用ReLU激活函数，定义了一个三层神经网络。
    支持传入用于计算参数均值的类，方便测试时使用滑动平均模型
'''
def inference(input_tensor,avg_class,
              weights1,biases1,weights2,biases2):
    
    #如果没有提供滑动平均类，使用参数当前值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        
        #此处不需要添加激活函数，计算损失函数时会一并计算softmax函数
        return tf.matmul(layer1,weights2)+biases2
    
    #提供了滑动平均类，则先计算滑动平均值，再计算神经网络前向传播结果
    #average函数用来调用对应参数的滑动平均值
    else:
        layer1 = tf.nn.relu(
                tf.matmul(input_tensor,avg_class.average(weights1)) +
                avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)) + \
                        avg_class.average(biases2)

#将模型的训练过程封装为一个函数
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    
    weights1 = tf.Variable(
            tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    
    weights2 = tf.Variable(
            tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    #计算神经网络的前向传播结果，使用inference函数
    y = inference(x,None,weights1,biases1,weights2,biases2)
    
    #定义用于存储训练轮数的变量，此变量不需要被训练
    global_step = tf.Variable(0,trainable=False)
    
    #给定平均衰减率和训练轮数，初始化滑动平均类
    #tf.train.ExponentialMovingAverage函数采用滑动平均的方法更新参数。
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY,global_step)
    #对所有可训练变量使用滑动平均
    variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
    
    """
    计算使用了滑动平均之后的前向传播结果
    滑动平均并不改变变量本身的取值，只是维护一个影子变量来记录平均值
    要使用这个平均值时，需调用average函数
    """
    average_y = inference(x,variable_averages,
                          weights1,biases1,weights2,biases2)
    
    #计算损失函数，sparse_softmax_cross_entropy_with_logits计算交叉熵
    #第一个参数是神经网络不包含softmax的前向传播结果
    #第二个参数是正确标签
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y,labels=tf.argmax(y_,1))
    #计算交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #计算L2正则化损失函数
    '''
    正则化可以帮助防止过拟合，提高模型的适用性。
    使用规则来使尽量少的变量去拟合数据，让模型无法完美匹配所有的训练项。
    L2正则化表达式暗示着一种倾向：训练尽可能的小的权重，
    较大的权重需要保证能显著降低原有损失C0才能保留。
    参数表示正则化权重。
    返回一个执行L2正则化的函数。
    '''
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算正则化损失，一般只计算矩阵部分，不计算偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失等于交叉熵和正则化损失的和，这样，模型复杂度本身会提高损失函数
    loss = cross_entropy_mean + regularization
    
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,global_step,
            mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    
    #优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).\
                    minimize(loss,global_step=global_step)
    
    #在训练时，每过一遍数据需要通过反向传播更新参数以及滑动平均值
    #以下两行代码用于流程控制，可以一次完成多个机制
    #也可使用train_op = tf.group(train_step,variables_averages_op)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    #判断正确性
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #计算正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    #初始化会话，开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        validate_feed = {x:mnist.validation.images,
                         y_:mnist.validation.labels}
        
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        
        for i in range(TRAINING_STEPS):
            if i % 300 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("经过%d次训练，正确率为%g" % (i,validate_acc))
            
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        #测试最终正确率
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("最终正确率为%g" % (test_acc))
        
def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    train(mnist)

#Tensorflow定义的主程序入口，会调用上面的main函数
if __name__ == '__main__':
    tf.app.run()
