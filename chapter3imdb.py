# データの準備
from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# !学習データと検証データのホールドアウト

# one-hot encoding
def vectorize_sequences(sequences, dimension=10000):
    # 計上が(len(sequences), dimension)の行列を作成し、0で埋める
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # results[i]のインデックスを１に設定
    return results
# !正規化プロセス

# train_dataのベクトル化
x_train = vectorize_sequences(train_data)
# x_testのベクトル化
x_test = vectorize_sequences(test_data)
# numpyを使った変換
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# !numpyを使った正規化プロセス


# ニューラルネットワーク層の構築
from keras import models
from keras import layers
# モデル定義
model = models.Sequential()
# !NNモデルのインスタンス
# 隠れ層（全結合層）入力10000、隠れ層16、出力1
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
# !全結合（ネットワーク）層の追加、入力層,活性化関数,ユニット数（入力データの形状と一致）
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# !出力層（微分可能でなければback propergationが働かない）


# モデルのコンパイル
from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              # オプティマイザ（最適化関数）
              loss=losses.binary_crossentropy,
              # 損失関数
              metrics=[metrics.binary_accuracy]
              # 評価指標
              )

# アプローチの検証
# trainデータを分割
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 学習プロセス
history = model.fit(partial_x_train,
                     partial_y_train,
                     epochs=4,
                     batch_size=512,
                     validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


