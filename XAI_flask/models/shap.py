import matplotlib.pyplot as pl
import shap
import numpy as np

import tensorflow as tf

from tensorflow.keras.datasets import mnist
from keras.models import load_model

class SHAP_MNIST:
    def __init__(self):
        model = load_model('./models/mnist_model.h5')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(x_test[1:5])

    def plot(self):
        shap.image_plot(self.shap_values, -self.x_test[1:5], show=False)
        pl.savefig('result.png')




# plot the feature attributions