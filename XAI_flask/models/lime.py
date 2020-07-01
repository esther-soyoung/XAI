from flask import __main__
from lime.lime_text import LimeTextExplainer
import dill
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
import numpy as np


def create_model(max_features):
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


class Padder(BaseEstimator, TransformerMixin):
    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None

    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
        return self

    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.maxlen)
        X[X > self.max_index] = 0
        return X


class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self

    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))


class LIME_NLP:
    def __init__(self):
        with open('models/pretrained/lime.pkl', 'rb') as f:
            __main__.TextsToSequences = TextsToSequences;
            self.pipeline = dill.load(f)

    def plot(self, review):
        y_preds = self.pipeline.predict(review)
        class_names = ['negative', 'positive']

        explainer = LimeTextExplainer(class_names=class_names)
        explanation = explainer.explain_instance(review, self.pipeline.predict_proba, num_features=10, top_labels=1)

        # explanation.show_in_notebook(text=review)

        html = explanation.as_html()
        return html
