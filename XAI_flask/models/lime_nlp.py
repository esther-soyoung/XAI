from lime.lime_text import LimeTextExplainer
from lime_nlp_utils import TextsToSequences, Padder, create_model
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os
import sys

df = pd.read_csv('models/pretrained/IMDB Dataset.csv')

texts_train, texts_test, y_train, y_test = \
    train_test_split(df["review"].values, df['sentiment'].values, random_state=42)

vocab_size = 20000
maxlen = 80

sequencer = TextsToSequences(num_words=vocab_size)
padder = Padder(maxlen)

with open('models/pretrained/lime.pkl', 'rb') as f:
    pipeline = pickle.load(f)

review = input("")


def lime_result(review):
    f = open(os.devnull, 'w')
    tmp_stdout = sys.stdout
    sys.stdout = f

    y_preds = pipeline.predict(review)
    class_names = ['negative', 'positive']

    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(review, pipeline.predict_proba, num_features=10, top_labels=1)
    result = explanation.as_html()

    sys.stdout = tmp_stdout
    return result


print(lime_result(review))
