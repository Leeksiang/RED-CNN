import numpy as np
import os
import tensorflow as tf
import datetime
from utils import load, save_images


def get_dirNums(path, current_time):
    nums = 0
    for file in os.listdir(path):
        if file.find(current_time) == 0:
            nums += 1
    return nums

def file_path(test_dir, current_dir, file):
    return os.path.join(test_dir, current_dir, '{0}_{1}'.format('outimg', os.path.basename(file).replace('L', 'H')))

def test(model, options):
    load(model.ckpt_manager, tf.train.Checkpoint(G=model.G))
    sample_files = 'mayo.txt'


    # write html for visual comparison
    current_time = datetime.datetime.today().strftime('%Y%m%d') #now().strftime("%Y%m%d-%H%M%S")
    current_dir = current_time + '_' + str(get_dirNums(options.test_dir, current_time))

    for file in open(sample_files):
        print('Processing image: ' + file)
        file = file.rstrip()
        sample_image = np.load(file)
        sample_image = tf.reshape(sample_image, [1, 512, 512, 1])
        image_path = file_path(options.test_dir, current_dir, file)

        print('image_path:', image_path)
        fake_img = model.G.predict(sample_image)
        print(fake_img.shape)
        save_images(fake_img, [1, 1], image_path.replace('npy', 'png'))
        fake_img = np.array(fake_img[0, :, :, 0])
        #np.save(os.path.join(options.test_dir, '{0}_{1}'.format('outimg' , os.path.basename(file).replace('L', 'H'))),fake_img)
        np.save(image_path, fake_img)


