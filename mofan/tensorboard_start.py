#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# @Time    : 18-7-26 下午4:59
# @Author  : viaeou
# @Site    : 
# @File    : tensorboard_start.py
# @Software: PyCharm


import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


with tf.name_scope('inputs'):
    # define placeholder for inputs to network
    train_x = tf.placeholder(tf.float32, shape=[None, 1], name='train_x')
    train_y = tf.placeholder(tf.float32, shape=[None, 1], name='train_y')


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(
                tf.random_normal([in_size, out_size]),
                name='w'
            )
        with tf.name_scope('bias'):
            biases = tf.Variable(
                tf.zeros([1, out_size]) + 0.1,
                name='b'
            )
        with tf.name_scope('add'):
            add_res = tf.add(tf.matmul(inputs, weights), biases)
        if activation_function is None:
            outputs = add_res
        else:
            outputs = activation_function(add_res, )
        return outputs


# add hidden layer
l1 = add_layer(train_x, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(train_y - prediction),
                                        reduction_indices=[1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)






# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs


