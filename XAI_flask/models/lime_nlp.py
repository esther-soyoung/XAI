# from lime.lime_text import LimeTextExplainer
# from lime_nlp_utils import TextsToSequences, Padder, create_model
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import pickle
#
# df = pd.read_csv('pretrained/IMDB Dataset.csv')
#
# texts_train, texts_test, y_train, y_test = \
#     train_test_split(df["review"].values, df['sentiment'].values, random_state=42)
#
# vocab_size = 20000
# maxlen = 80
#
# sequencer = TextsToSequences(num_words=vocab_size)
# padder = Padder(maxlen)
#
# with open('pretrained/lime.pkl', 'rb') as f:
#     pipeline = pickle.load(f)
#
# review = input("")
#
#
# def lime_result(review):
#     y_preds = pipeline.predict(review)
#     class_names = ['negative', 'positive']
#
#     explainer = LimeTextExplainer(class_names=class_names)
#     explanation = explainer.explain_instance(review, pipeline.predict_proba, num_features=10, top_labels=1)
#     explanation.as_html()
#
#
# print(lime_result(review))
import warnings

warnings.filterwarnings("ignore")
import ipywidgets as widgets
from ipywidgets import interact_manual

# 텐서플로 2 버전 선택

import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv('pretrained/IMDB Dataset.csv')
from sklearn.model_selection import train_test_split

texts_train, texts_test, y_train, y_test = \
    train_test_split(df["review"].values, df['sentiment'].values, random_state=42)

vocab_size = 20000  # Max number of different word, i.e. model input dimension
maxlen = 80  # Max number of words kept at the end of each text

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator

from lime_nlp_utils import TextsToSequences, Padder, create_model

sequencer = TextsToSequences(num_words=vocab_size)
padder = Padder(maxlen)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline

batch_size = 64
max_features = vocab_size + 1

import pickle

with open('pretrained/lime.pkl', 'rb') as f:
    pipeline = pickle.load(f)

review = input("글을 입력하세요 : ")
print()


def lime_result(review):
    y_preds = pipeline.predict(review)
    # print("예측 결과 : ", y_preds)
    class_names = ['negative', 'positive']
    from collections import OrderedDict
    from lime.lime_text import LimeTextExplainer
    from matplotlib import pyplot as plt

    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(review, pipeline.predict_proba, num_features=10, top_labels=1)
    explanation.as_html()


print(lime_result(review))
