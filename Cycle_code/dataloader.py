import numpy as np
import cv2
import os

class DataLoader():
    def __init__(self, A_data_path, B_data_path):
        self.A_data_arr = np.load(A_data_path)
        self.B_data_arr = np.load(B_data_path)
        self.load_counter = 0

    def load_data_cyclc(self, batch_size):
        A_images = self.A_data_arr[self.load_counter:self.load_counter+batch_size]
        B_images = self.B_data_arr[self.load_counter:self.load_counter+batch_size]

        self.load_counter += batch_size
        if(self.load_counter==len(self.A_data_arr)):
            self.load_counter = 0

        return A_images, B_images

    def data_num(self):
        return len(self.A_data_arr)

