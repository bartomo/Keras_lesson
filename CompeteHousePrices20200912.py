# from keras.datasets import boston_housing
import pickle

# (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#TODO load dataset from Compete

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

from keras import models
from keras import layers


def build_model(build_flag=True):
    """

    :param build_flag:
    :return:
    """
    model = models.Sequential()
    model.add(layers.Dense(64,
                           activation='relu',
                           input_shape=(train_data.shape[1],
                                        )))
    model.add(layers.Dense(64,
                           activation='relu',
                           ))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae']
                  )

    # TODO pickle save
    if build_flag:
        with open('compete_model.pkl', 'wb') as f:
            try:
                pickle.dump(model, f)
            except Exception as e:
                breakpoint()

    # TODO load pickle
    else:
        with open('compete_model.pkl', 'rb') as f:
            model = pickle.load(f)

    return model


# RMSE(二乗平均平方根誤差）
from keras.metrics import mean_squared_error

#  k分割交差検証
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    #  検証データの準備
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
    #  訓練データの準備
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i+1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i+1)*num_val_samples:]],
                                           axis=0)
    #  Kerasモデルを構築
    model = build_model()
    # pickle load
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    # all_scores.append(val_mae)
    # breakpoint()
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

