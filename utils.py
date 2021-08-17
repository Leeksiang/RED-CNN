"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import os
import pprint
import scipy.misc
import numpy as np
import copy
import tensorflow as tf
import scipy.io as sio
try:
    _imread = scipy.misc.imread
    # scipy.misc.imread（name, flatten=False, mode=None）
    # 读取图片为array
except AttributeError:
    from imageio import imread as _imread
# 提供了打印出任何python数据结构类和方法
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
# 图片池
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            # 随机抽取
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]  #浅复制
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load(ckpt_manager, ckpt):
    if ckpt and ckpt_manager._latest_checkpoint:
        ckpt.restore(ckpt_manager._latest_checkpoint).expect_partial()
        return True
    else:
        return False

def save(ckpt_manager, options):
    checkpoint_dir = os.path.join(options.checkpoint_dir, options.model_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckpt_save_path = ckpt_manager.save()
    print('checkpoint saved at ', ckpt_save_path)
    return ckpt_save_path

def read_tfrecord(serialized_example, img_size=128):
    feature_description = {
        'label':tf.io.FixedLenFeature((), tf.string),
        'data': tf.io.FixedLenFeature((), tf.string)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_raw(example['data'], tf.float32)
    image = tf.reshape(image, [img_size, img_size, 1])
    return image

def read_img(img, img_size=64):
    def _read_npy(img):
        img = np.load(img)
        return img

    out_img = tf.numpy_function(_read_npy, [img], [tf.float32])

    out_img = tf.reshape(out_img, [img_size, img_size, 1])
    return out_img


def read_mat(img, img_size=64):
    def _read_mat(img):
        img = sio.loadmat(img)
        img = img['imgsave']
        return img

    out_img = tf.numpy_function(_read_mat, [img], [tf.float32])

    out_img = tf.reshape(out_img, [img_size, img_size, 1])
    return out_img



def sample_model(model, options):
    model_name = options.model_name
    sample_dir = os.path.join(options.sample_dir, options.model_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    model.G.save(os.path.join(sample_dir, "G_" + model_name), save_format="tf")
    model.F.save(os.path.join(sample_dir, "F_" + model_name), save_format="tf")
    model.D_A.save(os.path.join(sample_dir, "D_A_" + model_name), save_format="tf")
    model.D_B.save(os.path.join(sample_dir, "D_B_" + model_name), save_format="tf")
    print('model saved at', sample_dir)



def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)








def get_dirNums(path):
    nums = 0
    for file in os.listdir(path):
        print(file)
        if (os.path.isdir(os.path.join(path, file))):
            nums += 1
    return nums
