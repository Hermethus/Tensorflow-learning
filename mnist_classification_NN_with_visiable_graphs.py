# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:29:18 2018

@author: Hermethus
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# 将One-Hot编码的向量转换成一个单独的数字
# data.test.labels二维向量 => data.test.cls一维向量
data.test.cls = np.array([label.argmax() for label in data.test.labels])


# 图像尺寸 基本参数
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10
batch_size = 100

# 用来绘制图像
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
 
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    # 调整图像间的空白间隔
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # enumerate将迭代器输出为元组序列 (下标(索引)，内容)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
 
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
 
        ax.set_xlabel(xlabel)

        # 设置坐标轴为不显示
        ax.set_xticks([])
        ax.set_yticks([])
'''
# Get the first images from the test-set.
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)
'''

# 用于进行优化
# num_iterations：训练次数
def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
 
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
    
    # 用于计算训练集的正确率，从batch中获取标签
    data.train.cls = np.array([label.argmax() for label in y_true_batch])
    acc = session.run(accuracy, feed_dict={x: x_batch,
                                           y_true: y_true_batch,
                                           y_true_cls: data.train.cls})
    print("Accuracy on train-set: {0:.1%}".format(acc),end='\t')


# 用测试集判断精确度
def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))
    

# 用scikit-learn打印混淆矩阵，数据集是测试集
def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = data.test.cls
 
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
 
    # 以文字方式打印混淆矩阵
    print(cm)
 
    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
 
    # 调整图像参数
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

# 绘制测试集中误分类的图像
def plot_example_errors():
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)
    # Negate the boolean array.
    incorrect = (correct == False)
 
    # 获得被错误分类的图像
    images = data.test.images[incorrect]
    # 获得错误分类
    cls_pred = cls_pred[incorrect]
    # 获得正确分类
    cls_true = data.test.cls[incorrect]
 
    # 打印前9个图像
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# 绘制模型权重，每个输出节点对应一张图
def plot_weights():
    # w是由权重构成的数组
    w = session.run(weights)
 
    # 获得权重中的最大最小值作为绘图的参考
    w_min = np.min(w)
    w_max = np.max(w)
 
    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
 
    for i, ax in enumerate(axes.flat):
        if i<10:
            # 将权重转化为图片
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)
 
            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
 
        # 设置坐标轴为不显示
        ax.set_xticks([])
        ax.set_yticks([])


'''=======================以下开始计算图的设置========================='''
# 设置占位符
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

# 设置模型变量
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))#[784,10]
biases = tf.Variable(tf.zeros([num_classes]))
logits = tf.matmul(x, weights) + biases

# 将判断结果从one-hot转为十进制数字
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# 计算交叉熵
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

# 计算损失函数
cost = tf.reduce_mean(cross_entropy)

# 设置优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# 计算正确率
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''=========================以上是计算图的设置==========================='''

# 构建会话
with tf.Session as session:
    
    session.run(tf.global_variables_initializer())
    
    # 构建测试集
    feed_dict_test = {x: data.test.images,
                      y_true: data.test.labels,
                      y_true_cls: data.test.cls}
    
    
    for i in range(10):
        optimize(num_iterations=1000)
        print_accuracy()
        
    plot_weights()
    print_confusion_matrix()

