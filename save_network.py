# -*- coding: utf-8 -*-
"""
Created on Sat May 12 22:49:17 2018

@author: Hermethus
"""

import tensorflow as tf

'''
保存和加载时两次定义了W = tf.Variable(xxx,name='weight')
第二个的实际 name 会变成 "weight_1"导致出错

在加载过程中，定义 name 相同的变量前面加 tf.reset_default_graph()
清除默认图的堆栈，并设置全局图为默认图
'''
tf.reset_default_graph()
ifSave = False

if ifSave:#保存神经网络
    
    #记得保证shape和dtype要相同
    W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
    b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess,'my_network/test.ckpt')
        print("Save to path:",save_path)

else:#读取神经网络
    
    #先定义一个空框架
    #记得保证shape和dtype要相同
    #这里是shape=(2,3),2行3列
    W = tf.Variable(tf.zeros([2,3]),dtype=tf.float32,name='weights')
    b = tf.Variable(tf.zeros([1,3]),dtype=tf.float32,name='biases')
    
    #读取时不用initial
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #读取之后会自动存入相同标识的变量
        saver.restore(sess,'my_network/test.ckpt')
        print("weights:",sess.run(W))
        print("biases:",sess.run(b))


