import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session

import math
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


class FilterViz():

    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.session)
        self._model = load_model('models/pretrained/mnist_model.h5')

        self._img_rows = 28
        self._img_cols = 28
        (self._x_train, y_train), (self._x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            self._x_train = self._x_train.reshape(self._x_train.shape[0], 1, self._img_rows, self._img_cols)
            self._x_test = self._x_test.reshape(self._x_test.shape[0], 1, self._img_rows, self._img_cols)
        else:
            self._x_train = self._x_train.reshape(self._x_train.shape[0], self._img_rows, self._img_cols, 1)
            self._x_test = self._x_test.reshape(self._x_test.shape[0], self._img_rows, self._img_cols, 1)

        self._x_train = self._x_train.astype('float32')
        self._x_test = self._x_test.astype('float32')
        self._x_train /= 255
        self._x_test /= 255

    def _get_hidden_layers(self, data):
        with self.graph.as_default():
            set_session(self.session)
            feature_extractor = tf.keras.Model(inputs=self._model.inputs,
                                               outputs=[layer.output for layer in self._model.layers[:-4]])
            image = tf.convert_to_tensor(np.expand_dims(data, axis=0))
            result = feature_extractor(image)
        return result

    def _plot_filter(self, feature):
        feature = K.eval(feature)
        # filters = feature.shape[3]
        filters = 18
        plt.figure(1, figsize=(8, 15))
        n_columns = 3
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(6, filters):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(feature[0, :, :, i], interpolation='nearest')
        img = BytesIO()
        plt.savefig(img, format='png', dpi=200)
        img.seek(0)
        return img

    '''
    data: (28, 28)
    layer range: 0, 1, 2, 3
    '''

    def get_FilterViz(self, data, layer):
        data = data.reshape(self._img_rows, self._img_cols, 1)
        data = data.astype('float32')
        data /= 255

        hiddens = self._get_hidden_layers(data)
        return self._plot_filter(hiddens[layer])
