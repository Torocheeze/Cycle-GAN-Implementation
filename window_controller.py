from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2
import numpy as np
from processor import Simple_processor
from UI import Ui_Imape_processor

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Imape_processor()
        self.ui.setupUi(self)
        self.setup_control()
        self.ori_image = None
        self.result = None

    def setup_control(self):
        self.ui.Import_Buttom.clicked.connect(self.open_file)
        self.ui.Transform_buttom.clicked.connect(self.trans_error)
        self.ui.Save_Buttom.clicked.connect(self.save_error)


    def display_img(self, image, in_or_out):
        height, width, channel = image.shape
        bytesPerline = 3 * width
        qimg = QImage(image, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        if(in_or_out):
            self.ui.Show_input.setPixmap(QPixmap.fromImage(qimg))
        else:
            self.ui.Show_Result.setPixmap(QPixmap.fromImage(qimg))

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        if(filename):
            self.ori_image = cv2.imread(filename, cv2.IMREAD_COLOR)
            self.ori_image = cv2.resize(self.ori_image, (256, 256), interpolation = cv2.INTER_CUBIC)
            self.display_img(self.ori_image, True)
            self.ui.Transform_buttom.clicked.disconnect(self.trans_error)
            self.ui.Transform_buttom.clicked.connect(self.trans)

    def trans(self):
        pro = Simple_processor(self.ori_image, './Cycle_code/data/20.h5')
        self.result = pro.pass_result()
        self.display_img(self.result, False)
        self.ui.Save_Buttom.clicked.disconnect(self.save_error)
        self.ui.Save_Buttom.clicked.connect(self.save_image)

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self,"save file","./", "JPEG files (*.jpg)")
        if(file_path):
            cv2.imwrite(file_path, self.result, [cv2.IMWRITE_JPEG_QUALITY, 90])


    def trans_error(self):
        reply = QMessageBox.critical(self, "Error", "Please import image first!", QMessageBox.Ok)
        return

    def save_error(self):
        reply = QMessageBox.critical(self, "Error", "Please import image and transform first!", QMessageBox.Ok)
        return

