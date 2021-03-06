from keras.preprocessing.image import ImageDataGenerator
from keras import models

train_dir = 'C:/Users/barto/PycharmProjects/KerasLessonPy37/cats_and_dogs_small/train'
validation_dir = 'C:/Users/barto/PycharmProjects/KerasLessonPy37/cats_and_dogs_small/validation'

# すべてのデータを1/255でスケーリング
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = test_datagen.flow_from_directory(train_dir,
                                                   target_size=(150, 150),
                                                   batch_size=20,
                                                   class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape', labels_batch.shape)
    break
