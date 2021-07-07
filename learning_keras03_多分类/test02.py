import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# Generate dummy data
x_train = np.random.random((1, 10, 10, 1))#1张10*10*1的图片
# 100张图片，每张100*100*3
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# 100*10
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
# 20*100

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.这适用于32个卷积滤波器，每个尺寸为3x3。
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd)

# model.fit(x_train, y_train, batch_size=32, epochs=10)
# score = model.evaluate(x_test, y_test, batch_size=32)
# model.save('/home/lzq/file/python/learning_keras03_多分类/model.h5')
print(x_train.shape)
print(type(x_train))
#print(x_train)

t = tf.convert_to_tensor(x_train, tf.float32)
y = keras.layers.Conv2D(5, (3, 3), activation='relu', input_shape=(10, 10, 1))(t)
print(y.shape)
y = keras.layers.Conv2D(2, (4, 4), strides=1)(y)
#print(y)
print(type(y))
print(y.shape)