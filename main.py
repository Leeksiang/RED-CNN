import os
from model import Network
from absl import app
from inference_npy import test
from config.config import  Option

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(_):

    options = Option()

    #when continue to train
    #options = options._replace(current_epoch=14, model_dir='20210424', continue_train=True)

    #when test
    #options = options._replace(model_dir='20210407', phase='test', img_size=512)
    #print(options.batch_idxs)

    model = Network(options)
    print('ready for train')
    model.train() #if options.phase == 'train' else test(model, options)

if __name__ == '__main__':
    app.run(main)