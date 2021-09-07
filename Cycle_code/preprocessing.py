import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class pre_processor():
    def __init__(self, input_path, npy_path, test_size, data_name, image_size=(128, 128)):
        self.input_path = input_path
        self.npy_path = npy_path
        self.test_size = test_size
        self.image_size = image_size
        self.data_name = data_name

    def load_and_split(self):
        file_list = os.listdir(self.input_path)

        images = []
        for img_path in file_list:
            image = cv2.imread(os.path.join(self.input_path, img_path), cv2.IMREAD_COLOR)
            print(np.shape(image))
            image = cv2.resize(image, self.image_size, interpolation = cv2.INTER_CUBIC)
            image = image[:, :, ::-1]
            image = image.astype(np.float)

            images.append(image)

        images = np.array(images)/127.5 - 1.
        data_train, data_test = train_test_split(images, test_size=self.test_size, shuffle=True)
        np.save(os.path.join(self.npy_path, self.data_name + "_train.npy"), data_train)
        np.save(os.path.join(self.npy_path, self.data_name + "_test.npy"), data_test)


if __name__ == '__main__':
    input_path = './data/color_nature/bw'
    save_npy_path = './data'
    test_size = 491
    data_name = 'bw'
    image_size = (128, 128)

    pre = pre_processor(input_path, save_npy_path, test_size, data_name, image_size=(128, 128))
    pre.load_and_split()
