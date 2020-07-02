from lime.lime_text import LimeTextExplainer
import pickle
from models.lime_nlp_utils import TextsToSequences, Padder, create_model

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator

from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline

class LIME_NLP:
    def __init__(self):
        None
        # # Loading a keras model
        # self.session = tf.Session()
        # self.graph = tf.get_default_graph()
        # set_session(self.session)
        #
        # with open('models/pretrained/lime.pkl', 'rb') as f:
        #     self.pipeline = pickle.load(f)

    def plot(self, review):
        # with self.graph.as_default():
        #     set_session(self.session)
        #     y_preds = self.pipeline.predict(review)

        with open('models/pretrained/lime.pkl', 'rb') as f:
            self.pipeline = pickle.load(f)
        class_names = ['negative', 'positive']

        explainer = LimeTextExplainer(class_names=class_names)

        explanation = explainer.explain_instance(review, self.pipeline.predict_proba, num_features=10, top_labels=1)

        html = explanation.as_html()
        return html
