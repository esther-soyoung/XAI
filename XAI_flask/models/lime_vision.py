import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from tensorflow.keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from io import BytesIO
import numpy as np
import json
import base64

img_rows, img_cols = 28, 28

num_classes = 10

# the data, split between train and test sets
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train /= 255
x_test /= 255

model = load_model('models/pretrained/mnist_model.h5')

explainer = lime_image.LimeImageExplainer()
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)


def predict_fn(images):
    return model.predict_proba(images[:, :, :, 0:1])


image = json.loads(input())

image = np.array(image)

image = image.astype('float64')
image /= 255

model_predict = model.predict_classes(image.reshape(1, 28, 28, 1))[0]
model_prob = model.predict(image.reshape(1, 28, 28, 1))[0]

explanation = (
    explainer
        .explain_instance(
        image, predict_fn,
        top_labels=10, hide_color=0, num_samples=100, segmentation_fn=segmenter)
)

plt.figure(figsize=(4, 4))
temp, mask = (
    explanation.get_image_and_mask(model_predict, positive_only=True, num_features=10, hide_rest=False)
)

plt.subplot(2, 2, 1)

plt.imshow(
    label2rgb(mask, temp, bg_label=0),
    interpolation='nearest')

plt.title('Positive Regions for {}'.format(model_predict))

# show all segments
temp, mask = (
    explanation.get_image_and_mask(model_predict, positive_only=False, num_features=10, hide_rest=False)
)

plt.subplot(2, 2, 2)

plt.imshow(
    label2rgb(4 - mask, temp, bg_label=0),
    interpolation='nearest')

plt.title('Positive/Negative Regions for {}'.format(model_predict))

# show image only
plt.subplot(2, 2, 3)
plt.imshow(temp, interpolation='nearest')
plt.title('Show output image only')

# show mask only
plt.subplot(2, 2, 4)
plt.imshow(mask, interpolation='nearest')
plt.title('Show mask only')

img = BytesIO()
plt.savefig(img, format='png', dpi=200)
plt.clf()
plt.cla()
plt.close()
img.seek(0)

print(base64.b64encode(img.getvalue()).decode('utf-8'), end='')
