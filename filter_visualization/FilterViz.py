import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import math
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

class FilterViz():

    def __init__(self, model):
        self._model = model
        #self._data_shape = data_shape
        #self._num_class = 10

    def _get_hidden_layers(self, data):
        feature_extractor = keras.Model(inputs=self._model.inputs, \
                            outputs=[layer.output for layer in self._model.layers[:-4]])
        image = tf.convert_to_tensor(np.expand_dims(data, axis=0))
        return feature_extractor(image)

    def _plot_filter(self, feature):
        feature = K.eval(feature)
        filters = feature.shape[3] 
        plt.figure(1, figsize=(20,20))
        n_columns = 5
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.imshow(feature[0,:,:,i], interpolation='nearest')
        img = BytesIO()
        plt.savefig(img, format = 'png', dpi = 500)
        img.seek(0)
        return img

    def get_FilterViz(self, data, layer):
        # reshape
        data = data.astype('float32')
        data /= 255

        hiddens = self._get_hidden_layers(data)
        print(hiddens[layer].shape)
        return self._plot_filter(hiddens[layer])
        # for layer in range(len(hiddens)):
        #     yield self._plot_filter(hiddens[layer])