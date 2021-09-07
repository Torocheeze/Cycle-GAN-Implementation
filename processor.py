import tensorflow as tf
from tensorflow.python.keras.models import Model
import numpy as np
import os
import cv2
from Cycle_code.cycle_resgen import res_generator
from PIL import Image as im

class Simple_processor():
    def __init__(self, image, model_path):

        self.model = res_generator((128, 128, 3))
        self.model.load_weights(model_path)

        self.image_pre = self.preprocess(image)

    def preprocess(self, ori_image):
        image = cv2.resize(ori_image, (128, 128), interpolation = cv2.INTER_CUBIC)
        image = image[:, :, ::-1]
        image = image.astype(np.float)

        image = np.array(image)/127.5 - 1.
        image = np.reshape(image, (1, 128, 128, 3))

        return image

    def pass_result(self):
        result_img = self.model.predict(self.image_pre)
        result_img = np.reshape(result_img, (128, 128, 3))

        result_img = (result_img+1)*127.5
        result_img = result_img[:, :, ::-1]
        result_img = cv2.resize(result_img, (256, 256), interpolation = cv2.INTER_CUBIC)

        ### Need to change data type back to display
        result_img = result_img.astype('uint8')


        return result_img