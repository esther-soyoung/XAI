import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.ops import gen_nn_ops
from tensorflow.keras.models import load_model
from io import BytesIO
import json
import base64


def getGradient(activation, weight, bias):
    W = tf.math.maximum(0., weight)
    b = tf.math.maximum(0., bias)
    z = tf.matmul(activation, W) + b

    dX = tf.matmul(1 / z, tf.transpose(W))

    return dX


def backprop_dense(activation, weight, bias, relevance):
    W = tf.math.maximum(0., weight)
    b = tf.math.maximum(0., bias)
    z = tf.matmul(activation, W) + b

    s = relevance / (z + 1e-10)
    c = tf.matmul(s, tf.transpose(W))

    return activation * c


def backprop_pooling(activation, relevance):
    z = MaxPool2D(pool_size=(2, 2))(activation)

    s = relevance / (z + 1e-10)
    c = gen_nn_ops.max_pool_grad_v2(orig_input=activation, orig_output=z, grad=s,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='VALID')

    return activation * c


def backprop_conv(activation, weight, bias, relevance):
    strides = (1, 1)
    W = tf.math.maximum(0., weight)
    b = tf.math.maximum(0., bias)

    layer = Conv2D(filters=W.shape[-1], kernel_size=(W.shape[0], W.shape[1]),
                   padding='VALID', activation='relu')

    layer.build(input_shape=activation.shape)

    layer.set_weights([W, b])

    z = layer(activation)

    s = relevance / (z + 1e-10)

    c = tf.compat.v1.nn.conv2d_backprop_input(activation.shape, W, s, [1, *strides, 1], padding='VALID')

    return activation * c


# data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 221, 251, 255, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 182, 237, 255, 255, 255, 250, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 244, 255, 255, 239, 179, 94, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 144, 255, 255, 204, 0, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 218, 255, 219, 0, 33, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 246, 255, 116, 66, 251, 255, 255, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 255, 60, 254, 255, 255, 248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 255, 253, 255, 246, 255, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 221, 255, 255, 255, 173, 255, 230, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 232, 128, 97, 255, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 138, 255, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 194, 255, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 255, 113, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 255, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 255, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 255, 255, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 255, 213, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 255, 169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 166, 255, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 234, 255, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 255, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 255, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 255, 251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 255, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 255, 246, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 115, 255, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 147, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
data = json.loads(input())

data = np.array(data)

data_shape = (1, 28, 28, 1)
data = data.astype('float32')
data /= 255
data_seg = tf.cast(data.reshape(data_shape), tf.float32)

model = load_model('models/pretrained/mnist_model.h5')
output_shapes = []
layers = []
weights = model.get_weights()[::2]
biases = model.get_weights()[1::2]

for e, i in enumerate(model.layers):
    if (("Conv" in str(i)) or ("MaxPooling" in str(i))
            or ("Flatten" in str(i)) or ("Dense" in str(i))):
        output_shapes.append(i.output_shape[1:])
        layers.append(i)

activations = []

a = data_seg
for layer in layers:
    a = layer(a)
    activations.append(a)

R = activations[-1]
wb_cnt = 0

for layer_num, layer in enumerate(layers[::-1]):

    if ("Flatten" in str(layer)):
        R = tf.reshape(R, (-1, *output_shapes[(~layer_num) - 1]))
    elif ("Dense" in str(layer)):
        a = activations[(~layer_num) - 1] if layer_num != (len(layers) - 1) else data_seg
        w = weights[~wb_cnt]
        b = biases[~wb_cnt]
        R = backprop_dense(a, w, b, R)
        wb_cnt += 1
    elif ("Conv" in str(layer)):
        a = activations[(~layer_num) - 1] if layer_num != (len(layers) - 1) else data_seg
        w = weights[~wb_cnt]
        b = biases[~wb_cnt]
        R = backprop_conv(a, w, b, R)
        wb_cnt += 1
    elif ("MaxPooling" in str(layer)):
        a = activations[(~layer_num) - 1] if layer_num != (len(layers) - 1) else data_seg
        R = backprop_pooling(a, R)

LRP_out = tf.reshape(tf.reduce_sum(R, axis=-1), data_shape[1:-1])

plt.imshow(LRP_out, cmap=plt.cm.jet)
plt.axis('off')
plt.title(np.argmax(activations[-1], axis=1))

img = BytesIO()
plt.savefig(img, format='png', dpi=500)
img.seek(0)

print(base64.b64encode(img.getvalue()).decode(), end='')
