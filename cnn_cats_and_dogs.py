from keras import layers
from keras import models
import pickle

# 畳み込みベース
# modelのインスタンス生成
model = models.Sequential()
# 全結合層=プーリング層＋畳み込み層
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
# dropout処理
model.add(layers.Dropout(0.5))
# ソフトマックス層
model.add(layers.Dense(512, activation='relu'))
# ？出力ユニット？出力する内容によって選択する必要があるか
model.add(layers.Dense(1, activation='sigmoid'))


# modelのコンパイル
from keras import optimizers

# 損失関数、オプティマイザ、指標関数の選択
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_dir = 'C:/Users/barto/PycharmProjects/KerasLessonPy37/cats_and_dogs_small/train'
validation_dir = 'C:/Users/barto/PycharmProjects/KerasLessonPy37/cats_and_dogs_small/validation'

# すべてのデータを1/255でスケーリング > データ拡張
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   )
# 検証データはそのまま
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = test_datagen.flow_from_directory(train_dir,
                                                   target_size=(150, 150),
                                                   batch_size=32,
                                                   class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape', labels_batch.shape)
    break

# 学習の実行
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)
# 学習済みモデルの保存
model.save('cats_and_dogs_small_1.h5')
breakpoint()

# 学習結果の保存
with open('history.pkl', 'wb') as f:
    try:
        pickle.dump(history, f)
    except Exception as e:
        breakpoint()

breakpoint()
# 学習済みモデルでの再開用
# model = models.load_model('cats_and_dogs_small_2.h5')
# breakpoint()
# with open('history.pkl', 'rb') as f:
#     history = pickle.load(f)
#
# breakpoint()

# 結果の表示
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 正解率をプロット
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acd')
plt.title('Training and validation acuracy')
plt.legend()

plt.figure()

# 損失値をプロット
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training')
plt.legend()

plt.show()
