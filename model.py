from __future__ import division
import time
import os
import tensorflow as tf
from tensorflow.keras import Model
from module.generator.generator_redcnn import Generator
from utils import read_img, load, save
from loss import generator_loss, perceptual_loss
from tensorflow.keras.optimizers import Adam
import datetime
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Network:

    def __init__(self, options):

        self.options = options

        # define network structure
        self.G = Generator(self.options.img_size) #

        #define generator optimizer
        self.G_optimizer = Adam(options.optimizer['lr'])

        self.ckpt = tf.train.Checkpoint(G=self.G)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       os.path.join(options.checkpoint_dir, options.model_dir),
                                                       max_to_keep=3,
                                                       checkpoint_name=options.model_name)
        if options.phase == 'train':
            #self.vgg = VGG19(include_top=False)
            #self.vgg.trainable = False
            self.summary_writer = tf.summary.create_file_writer('/log/' +
                                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

            self.step = 1
            self.lr = self.options.optimizer['lr']

            #print networks' trainable_variables
            print(self.G.summary())

    def train(self):
        """Train cyclegan"""


        start_time = time.time()
        print('train start: ' + str(self.step))
        # continue to train
        if self.options.continue_train:
            if load(self.ckpt_manager, self.ckpt):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        #print(self.options.low_ct_path)
        # get list of low-dose ct  and normal-dose ct respectively
        LDCT = tf.data.Dataset.list_files(self.options.low_ct_path, shuffle=False).\
            map(read_img, num_parallel_calls=4).repeat(self.options.epochs).batch(self.options.batch_size).prefetch(buffer_size=512)
        NDCT = tf.data.Dataset.list_files(self.options.normal_ct_path, shuffle=False).\
            map(read_img, num_parallel_calls=4).repeat(self.options.epochs).batch(self.options.batch_size).prefetch(buffer_size=512)

        LDCT = iter(LDCT)
        NDCT = iter(NDCT)

        for epoch in range(self.options.current_epoch, self.options.epochs):



            #for real_a, real_b in tf.data.Dataset.zip((LDCT, NDCT)):
            for idx in range(self.options.batch_idxs):

                real_a = next(LDCT)
                real_b = next(NDCT)

                g_loss, fake_b= self.train_step(real_a, real_b)

                with self.summary_writer.as_default():
                    tf.summary.scalar('g_loss', tf.reduce_mean(g_loss), step=epoch)


                    tf.summary.image('image/low', real_a, max_outputs=1, step=epoch)
                    tf.summary.image('image/generated', fake_b, max_outputs=1, step=epoch)
                    #tf.summary.image('image/cycle', cycle_a, max_outputs=1, step=epoch)
                    tf.summary.image('image/normal', real_b, max_outputs=1, step=epoch)
                    #for g in G_gradient:
                        #tf.summary.scalar('grad/G', tf.reduce_mean(g), step=epoch)
                # save sample per 100 steps
                if self.step % 100 == 0:
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                        epoch, idx, self.options.batch_idxs, time.time() - start_time)))
                    start_time = time.time()
                    print('g_loss={:.4f}'.format(g_loss))
                    print('======================')
                    # save this model per one thousand
                    if self.step % 1000 == 0:
                        # tf.py_function(model.ckpt_manager.save, [], [tf.string])
                        save(self.ckpt_manager, self.options)
                self.step += 1


    @tf.function
    def train_step(self, real_a, real_b):
        with tf.GradientTape(persistent=True) as tape:

            fake_b = self.G(real_a, training=True)

            g_loss = generator_loss(real_b, fake_b) #* self.options.weights['g_weight']

        # Calculate Generator 's Gradient
        G_gradient = tape.gradient(g_loss, self.G.trainable_variables)

        # Upgrade Generator's learning_rate
        self.G_optimizer.learning_rate = self.lr

        #Apply optimizer
        self.G_optimizer.apply_gradients(zip(G_gradient, self.G.trainable_variables))

        return g_loss, fake_b
