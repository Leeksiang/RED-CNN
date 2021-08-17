from __future__  import division
from utils import *
from ops import *
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose
from tensorflow_addons.layers import InstanceNormalization
"""
this module contains the blocks that used to compose cnn.
"""
class Encoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = conv2d(32, kernel_size=7, strides=1)
        self.conv2 = conv2d(64, kernel_size=3, strides=2, padding='same')
        self.conv3 = conv2d(128, kernel_size=3, strides=2, padding='same')
        self.bn = BatchNormalization()
        self.ins = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer='random_uniform', gamma_initializer='random_uniform')

    def call(self, inputs, training=True, mask=None):
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        x = tf.nn.relu(self.ins(self.conv1(x), name='in1'))
        x = tf.nn.relu(self.ins(self.conv2(x), name='in2'))
        x = tf.nn.relu(self.ins(self.conv3(x), name='in3'))
        return x

class Residual(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = conv2d(128, kernel_size=3, strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = conv2d(128, kernel_size=3, strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.ins1 = InstanceNormalization()
        self.ins2 = InstanceNormalization()

    def call(self, inputs, training=True, mask=None):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv1(x)
        x = self.ins1(x)
        x = tf.nn.relu(x)

        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv2(x)
        x = self.ins2(x)
        x = tf.add(x, inputs)
        return x

class Decoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same',
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = Conv2DTranspose(3, kernel_size=7, strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.ins1 = InstanceNormalization()
        self.ins2 = InstanceNormalization()
        self.ins3  = InstanceNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.ins1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.ins2(x)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        x = self.conv3(x)
        x = self.ins3(x)
        x = tf.nn.tanh(x)
        return x