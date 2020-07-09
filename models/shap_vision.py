import shap

import matplotlib.pyplot as plt
import numpy as np

from io import BytesIO

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model


class SHAP_MNIST:
    def __init__(self):
        model = load_model('models/pretrained/mnist_model.h5')

        self.img_rows, self.img_cols = 28, 28

        (self.x_train, y_train), (self.x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        background = self.x_train[np.random.choice(self.x_train.shape[0], 100, replace=False)]
        self.e = shap.DeepExplainer(model, background)

    def plot(self, data):
        data = data.reshape(1, self.img_rows, self.img_cols, 1)
        data = data.astype('float32')
        data /= 255

        shap_values = self.e.shap_values(data)
        shap.image_plot(shap_values, -data, show=False)

        img = BytesIO()
        plt.savefig(img, format='png', dpi=200)
        plt.clf()
        plt.cla()
        plt.close()
        img.seek(0)
        return img
