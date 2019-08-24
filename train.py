import os
import numpy as np
from PIL import Image
import csv

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

from tools import word2label

def get_images_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.bmp')]


def load_data():
    train_list = get_images_list('.\\train')
    test_list = get_images_list('.\\test')
    train_len = len(train_list)
    test_len = len(test_list)
    X_train = np.empty((train_len, 20, 20))
    X_test = np.empty((test_len, 20, 20))

    for i in range(train_len):
        img = Image.open(train_list[i])  # 打开图像
        X_train[i] = np.asarray(img, dtype='float64') / 256  # 将图像转化为数组并将像素转化到0-1之间
    X_train = X_train.reshape(-1, 20, 20, 1)

    for i in range(test_len):
        img = Image.open(test_list[i])  # 打开图像
        X_test[i] = np.asarray(img, dtype='float64') / 256  # 将图像转化为数组并将像素转化到0-1之间
    X_test = X_test.reshape(-1, 20, 20, 1)

    with open('.\\train\\data.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        Y_train = [row[1] for row in reader]

    with open('.\\test\\data.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        Y_test = [row[1] for row in reader]

    return X_train, X_test, Y_train, Y_test


def train(X_train, X_test, Y_train, Y_test):
    for i in range(len(Y_train)):
        Y_train[i] = word2label(Y_train[i])
    for i in range(len(Y_test)):
        Y_test[i] = word2label(Y_test[i])

    Y_train = keras.utils.to_categorical(Y_train, 28)
    Y_test = keras.utils.to_categorical(Y_test, 28)

    model = Sequential()
    model.add(Conv2D(32, (4, 4), activation='relu', input_shape=[20, 20, 1]))
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    batch_size = 100
    epochs = 50
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs)

    loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print('loss:%.4f accuracy:%.4f' % (loss, accuracy))
    model.save('Veri.h5')


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_data()
    train(X_train, X_test, Y_train, Y_test)
