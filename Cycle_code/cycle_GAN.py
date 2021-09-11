from __future__ import print_function, division
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Multiply, Concatenate, Add, Subtract
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from dataloader import DataLoader
from cycle_resgen import res_generator, discriminator

class Cycle_GAN():
    def __init__(self, image_size, batch_size, epochs, sample_interval, A_data_path, B_data_path, sample_save_path, model_save):
        self.image_d = image_size
        self.image_size = (image_size, image_size, 3)
        self.batch_size = batch_size
        self.epochs = epochs
        self.sample_interval = sample_interval
        self.A_data_path = A_data_path
        self.B_data_path = B_data_path
        self.save_sample = sample_save_path
        self.save_model = model_save

        if not os.path.isdir(self.save_sample):
            os.mkdir(self.save_sample)

        if not os.path.isdir(self.save_model):
            os.mkdir(self.save_model)


    def build_compute_graph(self, cycle_loss = 10.0):
        # Optimizer
        optimizer = Adam(0.0002, 0.5)

        # Build discriminators
        self.dis_A = discriminator(self.image_size)
        self.dis_B = discriminator(self.image_size)
        self.dis_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.dis_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        ### Build generator computational graph ###
        self.gen_A2B = res_generator(self.image_size)
        self.gen_B2A = res_generator(self.image_size)

        img_A = Input(self.image_size)
        img_B = Input(self.image_size)

        # Images change domain
        nise_B = self.gen_A2B(img_A)
        nise_A = self.gen_B2A(img_B)

        # images recovery
        recover_A = self.gen_B2A(nise_B)
        recover_B = self.gen_A2B(nise_A)

        # combined model only train the generators
        self.dis_A.trainable = False
        self.dis_B.trainable = False

        # discriminators decide images are real or not
        decide_A = self.dis_A(nise_A)
        decide_B = self.dis_B(nise_B)

        self.Comb_model = Model(inputs=[img_A, img_B], outputs=[decide_A, decide_B, recover_A, recover_B])
        self.Comb_model.compile(loss= ['mse', 'mse', 'mae', 'mae'], loss_weights= [1, 1, cycle_loss, cycle_loss], optimizer=optimizer)


    def training(self):
        loader = DataLoader(self.A_data_path, self.B_data_path)
        # patch
        patch = int(self.image_d / 2 ** 4)
        dis_patch = (patch, patch, 1)

        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        real = np.ones((self.batch_size,) + dis_patch)
        nise = np.zeros((self.batch_size,) + dis_patch)

        per_epoch = loader.data_num()//self.batch_size

        for count in range(self.epochs):
            for batch in range(per_epoch):
                img_A, img_B = loader.load_data_cyclc(self.batch_size)

                # Translate
                fake_B = self.gen_A2B.predict(img_A)
                fake_A = self.gen_B2A.predict(img_B)

                # Train discriminators
                dAloss_real = self.dis_A.train_on_batch(img_A, real)
                dAloss_fake = self.dis_A.train_on_batch(fake_A, nise)
                dA_loss = 0.5 * np.add(dAloss_real, dAloss_fake)

                dBloss_real = self.dis_B.train_on_batch(img_B, real)
                dBloss_fake = self.dis_B.train_on_batch(fake_B, nise)
                dB_loss = 0.5 * np.add(dBloss_real, dBloss_fake)

                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # Train generators
                g_loss = self.Comb_model.train_on_batch([img_A, img_B], [real, real, img_A, img_B])

                el_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f] time: %s " % (count, self.epochs, batch,
                per_epoch, d_loss[0], 100*d_loss[1], g_loss[0], el_time))


            if count % self.sample_interval == 0:
                for num_i in range(self.batch_size):
                    A = np.reshape(img_A[num_i], (1, self.image_d, self.image_d, 3))
                    B = np.reshape(img_B[num_i], (1, self.image_d, self.image_d, 3))
                    sample_A = self.gen_A2B.predict(A)
                    sample_B = self.gen_B2A.predict(B)

                    A = np.reshape(A, (self.image_d, self.image_d, 3))
                    B = np.reshape(B, (self.image_d, self.image_d, 3))
                    sample_A = np.reshape(sample_A, (self.image_d, self.image_d, 3))
                    sample_B = np.reshape(sample_B, (self.image_d, self.image_d, 3))

                    A = (A+1)*127.5
                    B = (B+1)*127.5
                    sample_A = (sample_A+1)*127.5
                    sample_B = (sample_B+1)*127.5

                    A = A[:, :, ::-1]
                    B = B[:, :, ::-1]
                    sample_A = sample_A[:, :, ::-1]
                    sample_B = sample_B[:, :, ::-1]

                    cv2.imwrite(os.path.join(self.save_sample, "Ori_A_" + str(count) + "_" + str(num_i) + ".jpg"), A, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cv2.imwrite(os.path.join(self.save_sample, "Ori_B_" + str(count) + "_" + str(num_i) + ".jpg"), B, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cv2.imwrite(os.path.join(self.save_sample, "Res_FromOri_A_" + str(count) + "_" + str(num_i) + ".jpg"), sample_A, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cv2.imwrite(os.path.join(self.save_sample, "Res_FromOri_B_" + str(count) + "_" + str(num_i) + ".jpg"), sample_B, [cv2.IMWRITE_JPEG_QUALITY, 90])

                self.gen_A2B.save_weights(os.path.join(self.save_model, str(count) + ".h5"))



if __name__ == '__main__':
    path_trainA = './data/bw_train.npy'
    path_trainB = './data/color_train.npy'

    res_path = './res_sample'
    res_model = './res_model'

    image_size = 128
    batch_size = 8
    epochs = 50
    sample_interval = 2

    cycle = Cycle_GAN(image_size, batch_size, epochs, sample_interval, path_trainA, path_trainB, res_path, res_model)
    cycle.build_compute_graph()
    cycle.training()