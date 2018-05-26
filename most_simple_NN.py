# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

#定义层与层之间的映射关系
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name = 'layer%s'% n_layer
    #可视化图层
    with tf.name_scope(layer_name):
        #定义权重和偏置
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            #用于记录变化
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
        #矩阵计算结果
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
        #输出是计算结果在激励函数之后的输出
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

#训练数据集
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#可视化图层，包括两个input
with tf.name_scope('inputs'):
    #占位符，用来接收训练数据
    #TensorFlow不能直接用数据集，而是用占位符构建计算图，再导入数据
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

#输入层到隐藏层，形状是[1,10]
l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
#隐藏层到输出层，形状是[10,1]
#in_size是隐藏层的out_size，out_size是y_data的size
prediction = add_layer(l1,10,1,n_layer=2,activation_function=None)

#计算损失函数,用均方差表示
#reduce_xxxx函数是对输入进行降维，通过求和或者求平均等方式
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
            tf.reduce_sum(
                    tf.square(ys - prediction),
                    reduction_indices=[1]
            )
           )
    #记录loss，很重要。手动管理，防止bug。
    loss_scalar = tf.summary.scalar('loss', loss)

#计算梯度，计算每个参数的步长变化，并且计算出新的参数值
#用最速下降法让交叉熵下降
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化，参数是Variable类型，需要初始化
init = tf.global_variables_initializer()
sess = tf.Session()

#可视化文件
#在上一级目录用>tensorboard --logdir logs --host=127.0.0.1调用
writer = tf.summary.FileWriter("logs/",sess.graph)

#以下内容是训练过程

sess.run(init)

'''
#图形化，打印数据集点
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
'''

#合并所有summary，tf.summary.merge_all()自动管理会出错
#merged = tf.summary.merge_all()

for _ in range(1001):
    #训练一步
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if (_ % 100 == 0):
        #记录结果
        #result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})#自动管理，出错
        
        #手动添加loss的记录
        loss_metall = sess.run(loss_scalar,feed_dict={xs:x_data,ys:y_data})
        #添加多条记录时使用数组
        writer.add_summary(loss_metall,_)
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        
'''
        #打印拟合红线，将前一条线消除再打印下一条线
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)
'''

