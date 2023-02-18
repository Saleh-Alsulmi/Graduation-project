from PyQt5 import QtWidgets, uic
import os
import numpy as np
from PyQt5 import QtWidgets
import pickle
import cv2
from features_extraction import LBP

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super(App,self).__init__()
        uic.loadUi('./ml_file.ui',self)
        self.setFixedSize(804,548)
        self.algorithm = 'SVM'
        self.radioButton.toggled.connect(lambda x:self.select_algorithm('SVM'))
        self.radioButton_2.toggled.connect(lambda x:self.select_algorithm('KNN'))
        self.radioButton_3.toggled.connect(lambda x:self.select_algorithm('NB'))
        self.pushButton.clicked.connect(self.select_image)
        self.pushButton_2.clicked.connect(self.get_prediction)
        self.show()
    def select_algorithm(self,name):
        self.algorithm = name
        print(name)
    def select_image(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self,
         'Import images', os.path.dirname(os.path.abspath(__file__)), 'file(*.jpg)')
        try:
            filepath=filename[0]
            encoded = filepath.encode("UTF-8")
            filepath = encoded.decode("UTF-8")
            print(filepath)
            if filepath!='':
                self.lineEdit.setText(filepath)
                self.pushButton_2.setEnabled(True)
                self.data = self.load_image(filepath)
        except Exception as e:
            pass
    def load_image(self,image_path):
        dim = (160,160)
        image= cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        image /= 255
        image = LBP(image)
        return image.flatten()
    def load_modul(self):
        if self.algorithm =='SVM':
            with open('./models/leukemia_SVM.pickle','rb' ) as model:
                self.model = pickle.load(model)
        elif self.algorithm =='NB':
            with open('./models/leukemia_NB.pickle','rb' ) as model:
                self.model = pickle.load(model)
        else:
            with open('./models/leukemia_KNN.pickle','rb' ) as model:
                self.model = pickle.load(model)
    def get_prediction(self):
        self.load_modul()

        if self.algorithm =='SVM':
            persentage = "100%"
        elif self.algorithm =='NB':
            persentage = "100%"
        else:
            persentage = "63.64%"

        inf = 'the sample is infected.\nalgorithem uesd:' + self.algorithm + "\n by persentage of: " + persentage
        uninf = 'the sample is not infected.\nalgorithem uesd:' + self.algorithm + "\n by persentage of: " + persentage

        result = self.model.predict([self.data])
        if result[0]==1:
            self.label_3.setText(inf)
        else:
            self.label_3.setText(uninf)