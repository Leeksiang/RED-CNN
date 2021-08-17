import tensorflow as tf
from tensorflow import keras
from criterion import abs_criterion
from ops import gram
from tensorflow.keras.layers import Concatenate

def generator_loss(real_img, generated_img):
    return tf.math.reduce_mean(tf.math.squared_difference(real_img, generated_img))

def perceptual_loss(vgg, real_image, fake_image):
    real_image_3c = Concatenate(axis=3)([real_image, real_image, real_image])
    fake_image_3c = Concatenate(axis=3)([fake_image, fake_image, fake_image])
    [w, h, d] = real_image_3c.get_shape().as_list()[1:]
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(vgg(fake_image_3c) - vgg(real_image_3c)))) / (w*h*d))


def content_loss(base, combination):
    return tf.reduce_mean(tf.square(combination - base))


