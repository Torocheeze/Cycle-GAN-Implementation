import os
import cv2
import numpy as np

def resize_rectangle_img(from_dir,to_dir,size):
    files = os.listdir(from_dir)

    for file in files:

        img = cv2.imread(from_dir+"\\"+file, 1)
        img = cv2.resize(img, (size,size))

        cv2.imwrite(to_dir+"\\"+file, img)

def BW(method):
    files = os.listdir('D:\\cycle_data\\color_nature\\color\\test')
    for file in files:
        print(file)
        if method == 0:
            save_canny_img('D:\\cycle_data\\color_nature\\color\\test' + "\\" +file,'D:\\cycle_data\\color_nature\\bw\\test' + "\\" +file)
        elif method == 1:
            gray('D:\\cycle_data\\color_nature\\color\\test' + "\\" +file,'D:\\cycle_data\\color_nature\\bw\\test' + "\\" +file)

def gray(from_path, des_path):
    img = cv2.imread(from_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(des_path, gray_img)

def save_canny_img(from_path,to_path):
    img = cv2.imread(from_path)
    canny_img = cv2.Canny(img, 200, 300,apertureSize=3)
    cv2.imwrite(to_path, canny_img)




if __name__ == '__main__':
    BW(1)