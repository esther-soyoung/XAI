from pdpbox import pdp, info_plots
import pandas as pd
import tensorflow as tf


class PDP_BOSTON:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz",
                                                                                          test_split=0.2)
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(y_train)
        x_test = pd.DataFrame(x_test)
        y_test = pd.DataFrame(y_test)
        cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        x = pd.concat([x_train, x_test])
        y = pd.concat([y_train, y_test])
        boston_data = x.copy()
        boston_data['MEDV'] = y
        boston_data.columns = cols

    def plot_basic(self, i):
        fig, axes, summary_df = info_plots.target_plot(
            df=boston_data, feature=cols[i], feature_name=cols[i], target='MEDV'
        )

