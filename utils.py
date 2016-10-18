import tensorflow as tf
import numpy as np


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def convert2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def le_net_conv_pool(input_image, filter_size, input_channel, output_channel):
    """One Convolution-Layer and one Max-Pooling-Layer"""
    with tf.name_scope('Convolution'):
        W_conv = weight_variable([filter_size, filter_size, input_channel, output_channel], 'W_conv')
        b_conv = bias_variable([output_channel], 'b_conv')
        h_conv = tf.nn.relu(convert2d(input_image, W_conv) + b_conv)
    with tf.name_scope('MaxPool'):
        h_pool = max_pool_2x2(h_conv)
    return h_pool


def save_model(sess, model_path):
    saver = tf.train.Saver()
    saver.save(sess, model_path)


def restore_model(sess, model_path):
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
