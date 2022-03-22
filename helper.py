import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import cv2
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ONE_HOT_ENCODING = 'OHE'
LABEL_ENCODING = 'LBE'


class Helper:
    ohe = OneHotEncoder()
    lbe = LabelEncoder()
    scaler = StandardScaler()

    def getEncodedY(self):
        encodedY = self.y
        if self.encoding == ONE_HOT_ENCODING:
            encodedY = self.ohe.fit_transform(np.reshape(self.y,(-1, 1))).toarray()
        elif self.encoding == LABEL_ENCODING:
            encodedY = self.lbe.fit_transform(self.y)
        return encodedY

    # encoding: ohe -> One hot encoding, lbe -> Label Encoding, None: No encoding
    def __init__(self, datasetPath='Data/features_30_sec.csv', enc=ONE_HOT_ENCODING):
        self.dataset = pd.read_csv(datasetPath)
        self.X = self.dataset.values[:, 2:-1]
        self.y = self.dataset.values[:, -1]
        self.encoding = enc
        self.X_train, self.X_test, self.y_train, self.y_test = ([], [], [], [])

    def getTrainTestNums(self, test_size=0.2):
        scaledX = self.scaler.fit_transform(np.array(self.X, dtype=float))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(scaledX, self.getEncodedY(),
                                                                                test_size=test_size, random_state=0)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def accuracy(self, y_pred):
        return accuracy_score(self.y_test, y_pred)

    def fScore(self, y_pred):
        return f1_score(self.y_test, y_pred, average='macro')

    def getImages(self, dirname='Data/images_original', img_size=256, test_size=0.2):
        maindir = os.listdir(dirname)
        self.X = []
        self.y = []
        for className in maindir:
            for imgName in os.listdir(dirname + "/" + className):
                img = cv2.imread(dirname + '/' + className + '/' + imgName)[..., ::-1]
                resized_arr = cv2.resize(img, (img_size, img_size))
                self.X.append(resized_arr)
                self.y.append(className)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.array(self.X), self.getEncodedY(),
                                                                                test_size=test_size, random_state=0)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def decodeY(self, encodedY):
        if self.encoding == ONE_HOT_ENCODING:
            return self.ohe.inverse_transform(encodedY)
        elif self.encoding == LABEL_ENCODING:
            return self.lbe.inverse_transform(encodedY)
        else:
            return encodedY

    def confusionMatrix(self, y_pred):
        y_true = self.y_test
        if self.encoding == ONE_HOT_ENCODING:
            y_pred = self.decodeY(y_pred)
            y_true = self.decodeY(y_true)

        labels = os.listdir('Data/images_original')
        cm1 = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm1, index=[i for i in labels],
                             columns=[i for i in labels])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap="RdPu")

    def evaluate(self, y_pred):
        print("Accuracy: ", self.accuracy(y_pred))
        print("F-Score: ", self.fScore(y_pred))
        self.confusionMatrix(y_pred)


def getLabels(datasetPath='Data/images_original'):
    return os.listdir(datasetPath)
