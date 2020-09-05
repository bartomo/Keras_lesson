from keras import layers
from keras import models
import pickle

# CNNをインスタンス化
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
#
# # modelのコンパイル
# from keras import optimizers
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])
#
# from keras.preprocessing.image import ImageDataGenerator
#
# train_dir = 'C:/Users/barto/PycharmProjects/KerasLessonPy37/cats_and_dogs_small/train'
# validation_dir = 'C:/Users/barto/PycharmProjects/KerasLessonPy37/cats_and_dogs_small/validation'
#
# # すべてのデータを1/255でスケーリング
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = test_datagen.flow_from_directory(train_dir,
#                                                    target_size=(150, 150),
#                                                    batch_size=20,
#                                                    class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(validation_dir,
#                                                         target_size=(150, 150),
#                                                         batch_size=20,
#                                                         class_mode='binary')
#
# for data_batch, labels_batch in train_generator:
#     print('data batch shape:', data_batch.shape)
#     print('labels batch shape', labels_batch.shape)
#     break
#
# history = model.fit_generator(train_generator,
#                               steps_per_epoch=100,
#                               epochs=30,
#                               validation_data=validation_generator,
#                               validation_steps=50)
# model.save('cats_and_dogs_small_1.h5')
#
# with open('history.pkl', 'wb') as f:
#     try:
#         pickle.dump(history, f)
#     except Exception as e:
#         breakpoint()
#
# breakpoint()
#
# model = models.load_model('cats_and_dogs_small_1.h5')
breakpoint()
with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

breakpoint()
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
