import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.ops import gen_nn_ops
import ipywidgets as widgets
from ipywidgets import interact_manual
import warnings
import copy
from io import BytesIO

# assert tf.__version__ == "2.1.0", "Tensorflow Version doesn't match"
warnings.filterwarnings("ignore")

class LRP():

    def __init__(self, model, data_shape):
        self.model = model
        self.data_shape = data_shape

        self.output_shapes = []
        self.layers = []
        self.weights = self.model.get_weights()[::2]
        self.biases = self.model.get_weights()[1::2]

        for e,i in enumerate(self.model.layers):
            if( ("Conv" in str(i)) or ("MaxPooling" in str(i))
                or ("Flatten" in str(i)) or ("Dense" in str(i)) ):
                self.output_shapes.append(i.output_shape[1:])
                self.layers.append(i)

    def getGradient(self, activation, weight, bias):
        W = tf.math.maximum(0., weight)
        b = tf.math.maximum(0., bias)
        z = tf.matmul(activation, W) + b

        dX = tf.matmul(1 / z, tf.transpose(W))

        return dX

    def backprop_dense(self, activation, weight, bias, relevance):
        W = tf.math.maximum(0., weight)
        b = tf.math.maximum(0., bias)
        z = tf.matmul(activation, W) + b

        s = relevance / (z + 1e-10)
        c = tf.matmul(s, tf.transpose(W))

        return activation * c

    def backprop_pooling(self, activation, relevance):

        z = MaxPool2D(pool_size=(2, 2))(activation)

        s = relevance / (z + 1e-10)
        c = gen_nn_ops.max_pool_grad_v2(orig_input=activation, orig_output=z, grad=s,
                                        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        return activation * c

    def backprop_conv(self, activation, weight, bias, relevance):
        strides = (1, 1)
        W = tf.math.maximum(0., weight)
        b = tf.math.maximum(0., bias)

        layer = Conv2D(filters=W.shape[-1], kernel_size=(W.shape[0], W.shape[1]),
                       padding="SAME", activation='relu')

        layer.build(input_shape=activation.shape)

        layer.set_weights([W, b])

        z = layer(activation)

        s = relevance / (z + 1e-10)

        c = tf.compat.v1.nn.conv2d_backprop_input(activation.shape, W, s, [1, *strides, 1], padding='SAME')

        return activation * c

    def get_LRP(self, data):
        data_seg = tf.cast(data.reshape(self.data_shape), tf.float32)

        self.activations = []

        a = copy.deepcopy(data_seg)
        for layer in self.layers:
            a = layer(a)
            self.activations.append(a)

        R = self.activations[-1]
        wb_cnt = 0

        for layer_num, layer in enumerate(self.layers[::-1]):

            if("Flatten" in str(layer)):
                R = tf.reshape(R, (-1, *self.output_shapes[(~layer_num)-1]))
            elif("Dense" in str(layer)):
                a = self.activations[(~layer_num) - 1] if layer_num != (len(self.layers) - 1) else data_seg
                w = self.weights[~wb_cnt]
                b = self.biases[~wb_cnt]
                R = self.backprop_dense(a, w, b, R)
                wb_cnt += 1
            elif("Conv" in str(layer)):
                a = self.activations[(~layer_num) - 1] if layer_num != (len(self.layers) - 1) else data_seg
                w = self.weights[~wb_cnt]
                b = self.biases[~wb_cnt]
                R = self.backprop_conv(a, w, b, R)
                wb_cnt += 1
            elif("MaxPooling" in str(layer)):
                a = self.activations[(~layer_num) - 1] if layer_num != (len(self.layers) - 1) else data_seg
                R = self.backprop_pooling(a, R)

        LRP_out = tf.reshape(tf.reduce_sum(R, axis = -1), self.data_shape[1:-1])

        plt.imshow(LRP_out, cmap = plt.cm.jet)
        plt.axis('off')
        plt.title(np.argmax(self.activations[-1], axis = 1))

        img = BytesIO()
        plt.savefig(img, format = 'png', dpi = 500)
        img.seek(0)
        return img