from pdpbox import pdp, info_plots
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import json
import base64


class PDP_BOSTON:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz",
                                                                                          test_split=0.2)
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(y_train)
        x_test = pd.DataFrame(x_test)
        y_test = pd.DataFrame(y_test)
        self.cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                     'MEDV']
        x = pd.concat([x_train, x_test])
        y = pd.concat([y_train, y_test])
        self.boston_data = x.copy()
        self.boston_data['MEDV'] = y
        self.boston_data.columns = self.cols

    def plot(self, i):
        fig, _, _ = info_plots.target_plot(
            df=self.boston_data, feature=self.cols[i], feature_name=self.cols[i], target='MEDV'
        )

        img = BytesIO()
        plt.savefig(img, format='png', dpi=200)
        plt.clf()
        plt.cla()
        plt.close()

        img.seek(0)

        return img
