import tensorflow as tf
from tensorflow import keras
from utils import *
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn

def conv2d(filters, kernel_size, strides=1, padding='valid', kernel_initializer=tf.random_normal_initializer(stddev=0.02)):
    keras.layers.Conv2D(filters, kernel_size, strides, padding, kernel_initializer=kernel_initializer)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def gram(features):
    # _, h, w, c = features.get_shape()
    # features_reshape = tf.reshape(features, (-1, c))
    # return tf.matmul(tf.transpose(features_reshape), features_reshape) / (h * w * c)
    result = tf.linalg.einsum('bijc, bijd -> bcd', features, features)
    shape = tf.shape(features)
    num_locations = tf.cast(shape[1] * shape[2], tf.float32)
    return result / num_locations

def sobel(image):
    static_image_shape = image.get_shape()
    image_shape = array_ops.shape(image)
    kernels = [[[-1, -2, -1],[0, 0, 0], [1, 2, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
               [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
               [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]]
    num_kernels = len(kernels)
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2)
    kernels_tf = constant_op.constant(kernels, dtype= tf.float32)
    pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
    padded = array_ops.pad(image, pad_sizes, mode='REFLECT')
    strides = [1, 1, 1, 1]
    output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
    shape = array_ops.concat([image_shape, [num_kernels]], 0)
    output = array_ops.reshape(output, shape=shape)
    output.set_shape(static_image_shape.concatenate([num_kernels]))
    return output

def sn_weight(w, num_iters=1, update_collection=None):
    w_shape= w.shape.as_list()
    w_r = tf.reshape(w, [-1, w_shape[-1]]) # [-1, output_channel]
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.initializers.truncated_normal(), trainable=False)