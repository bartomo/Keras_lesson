from keras import models
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from keras.datasets import boston_housing
import pickle
from sklearn.model_selection import train_test_split

# TODO 評価指標の選定　RMSE(二乗平均平方根誤差）
from keras.metrics import mean_squared_error


# TODO load dataset from Compete
## テストデータの準備
# (train_df, train_targets), (test_df, test_targets) = boston_housing.load_data()
data_dir = 'Compete/house-prices-advanced-regression-techniques/'
train_csv = 'train.csv'
test_csv = 'test.csv'

# TODO csvファイル読み込み
train_df = pd.read_csv(data_dir + train_csv, index_col=0)
test_df = pd.read_csv(data_dir + test_csv, index_col=0)
breakpoint()

# TODO データの成型
# 要らないcolum除去

# TODO ワンホットエンコーディング
# カテゴライズ pandas.get_dummies()
train_df = pd.get_dummies(train_df)
breakpoint()

# TODO データの分割 目的変数SalePriceの分割
# X_train, X_test, Y_train, Y_test = train_test_split(train_df, test_df, train_size=0.2)
# train_data, train_targets, test_train, test_targets = train_test_split(train_df, test_df, train_size=0.2)
train_data = train_df.drop('SalePrice', axis=1)  # , inplace=True)
train_targets = train_df['SalePrice']

breakpoint()

# TODO データの正規化
mean = train_df.mean(axis=0)
train_df -= mean
std = train_df.std(axis=0)
train_df /= std

test_df -= mean
test_df /= std

# TODO 欠損値処理

# TODO 異常値、外れ値の除去または補正


# TODO CNNモデル構築
def build_model(build_flag=True):
    """

    :param build_flag:
    :return:
    """
    model = models.Sequential()
    model.add(layers.Dense(64,
                           activation='relu',
                           input_shape=(train_df.shape[1],
                                        )))
    model.add(layers.Dense(64,
                           activation='relu',
                           ))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae']
                  )

    # pickle save
    if build_flag:
        with open('compete_model.pkl', 'wb') as f:
            try:
                pickle.dump(model, f)
            except Exception as e:
                breakpoint()

    # load mode
    else:
        with open('compete_model.pkl', 'rb') as f:
            model = pickle.load(f)

    return model


#  k分割交差検証

k = 4
num_val_samples = len(train_df) // k
num_epochs = 500
all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    #  検証データの準備
    val_data = train_df[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
    #  訓練データの準備
    partial_train_df = np.concatenate([train_df[:i * num_val_samples],
                                         train_df[(i+1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i+1)*num_val_samples:]],
                                           axis=0)
    #  Kerasモデルを構築
    model = build_model(build_flag=False)
    # pickle load
    history = model.fit(partial_train_df, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

print(all_scores)
# TODO 学習済みモデルの保存

with open('compete_model_fit.pkl', 'wb') as f:
    try:
        pickle.dump(model, f)
    except Exception as e:
        print(e)
        breakpoint()

# load mode
with open('compete_model_fit.pkl', 'rb') as f:
    model = pickle.load(f)

breakpoint()

# TODO plotチェック
