import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import os
import platform
import matplotlib.pyplot as plt

print(tf.__version__)
print(np.__version__)

base_path = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting'
sentarus_dataset = r'sentaurus_ml_db.csv'

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

if __name__ == "__main__":
    if platform.system() == 'Windows':
        base_path = r'\\?\\' + os.path.abspath(base_path)

    df = pd.read_csv(os.path.join(base_path, sentarus_dataset))
    # If fitting pmpp uncomment the next line
    column_list = list(set(list(df.columns)) - set(['Rsh (Ohms cm2)', 'time (s)']))
    # If fitting rsh uncomment the next line
    # column_list = list(set(list(df.columns)) - set(['pd_mpp (mW/cm2)', 'time (s)']))
    column_list.sort()
    df = df[column_list]
    print(df.tail)
    # print(df.describe())

    # # If fitting pmpp uncomment the next line
    target_column = ['pd_mpp (mW/cm2)']
    # If fitting rsh uncomment the next line
    # target_column = ['Rsh (Ohms cm2)']
    predictors = list(set(list(df.columns)) - set(target_column))
    # df[predictors] = df[predictors] / df[predictors].max()
    # print(df.describe())

    X = df[predictors].values
    # y = df[target_column].values

    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('pd_mpp (mW/cm2)')
    test_labels = test_features.pop('pd_mpp (mW/cm2)')

    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))

    first = np.array(train_features[:1])

    # with np.printoptions(precision=2, suppress=True):
    #     print('First example:', first)
    #     print()
    #     print('Normalized:', normalizer(first).numpy())

    # mpp = np.array(train_features['pd_mpp (mW/cm2)'])

    mpp_normalizer = preprocessing.Normalization(input_shape=[X.shape[1], ])
    mpp_normalizer.adapt(np.array(train_features))

    # mpp_model = tf.keras.Sequential([
    #     mpp_normalizer,
    #     layers.Dense(units=1)
    # ])

    # mpp_model.summary()

    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(X.shape[1] * 10, activation='relu'),
            layers.Dense(X.shape[1] * 10, activation='relu'),
            layers.Dense(X.shape[1] * 10, activation='relu'),
            layers.Dense(X.shape[1] * 10, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.01))
        return model

    mpp_model = build_and_compile_model(mpp_normalizer)
    mpp_model.summary()

    history = mpp_model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=0, epochs=100
    )

    plot_loss(history)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    test_results = {}
    test_results['mpp_dnn_model'] = mpp_model.evaluate(test_features[predictors], test_features, verbose=0)
    print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)
    plt.show()
    model_save_path = os.path.join(base_path, 'dnn_mpp.h5')
    mpp_model.save(model_save_path, save_format='h5')

