import tensorflow as tf
import random
import os
from absl import app
from scipy.io import loadmat as load
import numpy as np
from numpy import float64
os.environ['CUDA_VISIBLE_DEVICES']='0'
from os import scandir


X_input_dir = 'trainA.txt'
Y_input_dir = 'trainB.txt'
X_output_file = 'data/tfrecords/low_128.tfrecords'
Y_output_file = 'data/tfrecords/normal_128.tfrecords'



def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
  input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
  file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.npy') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths
  
def data_reader1(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
  input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
  file_paths: list of strings
  """
  file_paths = []

  for file in open(input_dir):
    line = file.rstrip()
    filename = os.path.basename(line)
    file_paths.append(line)
    '''
    if filename[0:3] != '10':
      file_paths.append(line)
    else:
      print(filename[0:3])

    if img_file.name.endswith('.mat') and img_file.is_file():
      file_paths.append(img_file.path)
    '''
  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
def _floats_feature(value):
  """Wrapper for inserting floats features into Example proto."""
  return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def _convert_to_example(file_path, image_buffer):
  """Build an Example proto for an example.
  Args:
  file_path: string, path to an image file, e.g., '/path/to/example.JPG'
  image_buffer: string, JPEG encoding of RGB image
  Returns:
  Example proto
  """
  file_name = file_path.split('/')[-2]
  #print(file_name)

  example = tf.train.Example(features=tf.train.Features(feature={
    'label': _bytes_feature(tf.compat.as_bytes(file_name)),
    'data': _bytes_feature(image_buffer)
  }))
  return example

def data_writer(input_dir, output_file):
  """Write data to tfrecords

  """
  file_paths = data_reader1(input_dir)

  # create tfrecords dir if not exists
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except os.error as e:
    pass

  images_num = len(file_paths)

  # dump to tfrecords file
  writer = tf.io.TFRecordWriter(output_file)

  for i in range(len(file_paths)):
    file_path = file_paths[i]

    data = np.load(file_path).astype(np.float32).tostring()
    example = _convert_to_example(file_path, data)
    writer.write(example.SerializeToString())

    if i % 10 == 0:
      print("Processed {}/{}.".format(i, images_num))
  print("Done.")
  writer.close()

def main(unused_argv):
  print("Convert X data to tfrecords...")
  data_writer(X_input_dir, X_output_file)
  print("Convert Y data to tfrecords...")
  data_writer(Y_input_dir, Y_output_file)

if __name__ == '__main__':
  app.run(main)