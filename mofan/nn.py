#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# @Time    : 18-7-26 下午2:46
# @Author  : viaeou
# @Site    : 
# @File    : nn.py
# @Software: PyCharm


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


x_data = np.random.rand(300, 2)
y_data = np.random.randint(0, 2, size=(300, 1), dtype=np.int)

xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


l1 = add_layer(xs, 2, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=tf.sigmoid)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(10000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    if i % 1000 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
