import shap

import matplotlib.pyplot as pl
import numpy as np

from io import BytesIO, StringIO

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

class SHAP_MNIST:
    def __init__(self):
        model = load_model('models/mnist_model.h5')

        img_rows, img_cols = 28, 28

        (self.x_train, y_train), (self.x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, img_rows, img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        background = self.x_train[np.random.choice(self.x_train.shape[0], 100, replace=False)]
        e = shap.DeepExplainer(model, background)
        self.shap_values = e.shap_values(self.x_test[1:5])

    def plot(self):
        shap.image_plot(self.shap_values, -self.x_test[1:5], show=False)
        img = BytesIO()
        pl.savefig(img, format='png', dpi=500)
        img.seek(0)
        return img




# plot the feature attributions