import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.preprocessing import sequence
from tensorflow.compat.v1.keras.layers import Dense, Embedding, LSTM
from tensorflow.compat.v1.keras.preprocessing.text import text_to_word_sequence
from tensorflow.compat.v1.keras.models import load_model,Sequential
from tensorflow.compat.v1.keras.datasets import imdb
from tensorflow.compat.v1.keras.backend import set_session

import numpy as np
import shap
from io import BytesIO
import matplotlib.pyplot as plt

class SHAP_NLP:
    def __init__(self):
        (self.x_train, _), (_, _) = imdb.load_data(num_words=20000)

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=80)

        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.session)
        self.model = load_model('models/pretrained/shap_imdb.h5')

    def plot(self, review):
        result = text_to_word_sequence(review)

        words = imdb.get_word_index()

        result = list(filter(lambda x: x in words and int(words.get(x)) <= 20000, result))

        preprocess = np.array([list(map(lambda x: int(words.get(x)), result))])
        preprocess = sequence.pad_sequences(preprocess, maxlen=80)
        # we use the first 100 training examples as our background dataset to integrate over

        with self.graph.as_default():
            set_session(self.session)
            explainer = shap.DeepExplainer(self.model, self.x_train[:100])

        # explain the first 10 predictions
        # explaining each prediction requires 2 * background dataset size runs
        with self.graph.as_default():
            set_session(self.session)
            shap_values = explainer.shap_values(preprocess)
        # init the JS visualization code
        shap.initjs()

        # transform the indexes to words
        words = imdb.get_word_index()
        num2word = {}
        for w in words.keys():
            num2word[words[w]] = w

        x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), preprocess[0])))])

        # plot the explanation of the first prediction
        # Note the model is "multi-output" because it is rank-2 but only has one column
        shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test_words[0], matplotlib=True, show=False)

        img = BytesIO()
        plt.savefig(img, format='png', dpi=200)
        plt.clf()
        plt.cla()
        plt.close()
        img.seek(0)
        return img
