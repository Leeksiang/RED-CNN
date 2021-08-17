import numpy as np
import os
import tensorflow as tf
import datetime
import scipy.io as sio
from utils import load, save_images
os.environ['CUDA_VISIBLE_DEVICES']='0'

def test(model, options):
	load(model.ckpt_manager, tf.train.Checkpoint(G=model.G))
	sample_files = 'test_A.txt'

	# write html for visual comparison
	index_path = os.path.join(options.test_dir, '{0}_index.html'.format(options.which_direction))
	index = open(index_path, "w")
	index.write("<html><body><table><tr>")
	index.write("<th>name</th><th>input</th><th>output</th></tr>")

	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	for file in open(sample_files):
		print('Processing image: ' + file)
		file = file.rstrip()
		mat_in = sio.loadmat(file)['imgsave']
		#sample_image = np.array(mat_in).reshape(1, 512, 512, 1).astype(np.float32)
		mat_in = tf.cast(mat_in, dtype=tf.float32)
		sample_image = tf.reshape(mat_in, [1, 512, 512, 1])

		image_path = os.path.join(options.test_dir,
								  current_time,
								  '{0}_{1}'.format(options.which_direction,
												   os.path.basename(file).replace('mat', 'png')))
		print('image_path:', image_path)
		fake_img = model.G.predict(sample_image)
		print(fake_img.shape)
		save_images(fake_img, [1, 1], image_path)
		fake_img = np.array(fake_img[0, :, :, 0])
		sio.savemat(options.test_dir + '/outimg_' + os.path.basename(file), {'out_img':fake_img})
		sio.savemat(os.path.join(options.test_dir, current_time,
							 '{0}_{1}_{2}'.format('outimg', options.which_direction, os.path.basename(file))),
					{'out_img':fake_img})

		index.write("<td>%s</td>" % os.path.basename(image_path))
		index.write("<td><img src='%s'></td>" % (file if os.path.isabs(file) else (
				'..' + os.path.sep + file)))
		index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
				'..' + os.path.sep + image_path)))
		index.write("</tr>")
	index.close()
