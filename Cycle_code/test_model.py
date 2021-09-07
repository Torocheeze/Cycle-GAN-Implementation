import tensorflow as tf
from tensorflow.python.keras.models import Model
import numpy as np
import os
import cv2
from dataloader import DataLoader
from cycle_resgen import res_generator, discriminator

class tester():
    def __init__(self, image_size, test_path, model_path, res_path):
        self.image_size = image_size
        self.test_path = test_path
        self.model = res_generator((image_size, image_size, 3))
        self.model.load_weights(model_path)
        self.res_path = res_path

    def preprocess(self):
        file_list = os.listdir(self.test_path)

        images = []
        for img_path in file_list:
            image = cv2.imread(os.path.join(self.test_path, img_path), cv2.IMREAD_COLOR)
            print(np.shape(image))
            image = cv2.resize(image, self.image_size, interpolation = cv2.INTER_CUBIC)
            image = image[:, :, ::-1]
            image = image.astype(np.float)

            images.append(image)

        images = np.array(images)/127.5 - 1.

        return images

    def testing(self, raw_or_not=False):
        if(raw_or_not):
            data_arr = self.preprocess()
        else:
            data_arr = np.load(self.test_path)

        count = 0
        for image in data_arr:
            A = np.reshape(image, (1, self.image_size, self.image_size, 3))
            sample_A = self.model.predict(A)

            A = np.reshape(A, (self.image_size, self.image_size, 3))
            sample_A = np.reshape(sample_A, (self.image_size, self.image_size, 3))

            A = (A+1)*127.5
            sample_A = (sample_A+1)*127.5

            A = A[:, :, ::-1]
            sample_A = sample_A[:, :, ::-1]

            cv2.imwrite(os.path.join(self.res_path, "Ori_A_" + str(count) + ".jpg"), A, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(os.path.join(self.res_path, "test_FromOri_A_" + str(count) + ".jpg"), sample_A, [cv2.IMWRITE_JPEG_QUALITY, 90])
            count+=1


if __name__ == '__main__':
    img_size = 128
    test_path = './data/bw_test.npy'
    model_path = './data/20.h5'
    res_path = './test_res'

    t = tester(img_size, test_path, model_path, res_path)
    t.testing()




