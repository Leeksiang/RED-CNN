import datetime
from collections import namedtuple
from glob import glob
import os



def Option():

    low_ct_path = '/low/*.npy'
    normal_ct_path = '/normal/*.npy'

    model_name = 'red-cnn'
    epochs = 20  # of epoch
    epoch_step = 10  # of epoch to decay lr
    batch_size = 128  # images in batch
    current_epoch = 0 # current epoch of train

    checkpoint_dir = '/state/checkpoint'  # trained models are saved here
    sample_dir = '/state/sample'
    test_dir = './test'
    # calculate size of batch
    lc = glob(low_ct_path)
    nc = glob(normal_ct_path)
    batch_idxs = min(len(lc), len(nc)) // batch_size
    optimizer = {'lr': 1e-4}  # initial learning rate and momentum term of adam
    weights = {'g_weight': 1.0, 'content_weight': 0.15}  # weight on L1 term in objective
    model_dir = datetime.date.today().strftime('%Y%m%d')
    continue_train = False # if continue training, load the latest model
    phase = 'train'  # function was used to train or test
    img_size = 64  # of image size used to train


    OPTIONS = namedtuple('OPTIONS', ['low_ct_path', 'normal_ct_path', 'model_name', 'epochs', 'epoch_step',
                                     'batch_size', 'current_epoch',
                                     'checkpoint_dir', 'sample_dir', 'model_dir', 'test_dir',
                                     'optimizer', 'weights', 'continue_train', 'phase', 'img_size', 'batch_idxs'])
    options = OPTIONS._make([low_ct_path, normal_ct_path, model_name, epochs, epoch_step,
                             batch_size, current_epoch,
                             checkpoint_dir, sample_dir, model_dir, test_dir,
                             optimizer, weights, continue_train, phase, img_size, batch_idxs])

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    return options