from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np


# classify as flying object or non flying
# 0 - airplane
# 1 - automobile
# 2 - bird
# 3 - cat
# 4 - deer
# 5 - dog
# 6 - frog
# 7 - horse
# 8 - ship
# 9 - truck


def select_flying_objects(data):
    for i in range(0, len(data)):
        if data[i] == 0 or data[i] == 2:
            data[i] = 0
        else:
            data[i] = 1
    return data


if __name__ == '__main__':
    # 1 hidden layer
    model1 = models.Sequential()
    model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model1.add(layers.MaxPooling2D((2, 2)))
    model1.add(layers.Flatten())
    model1.add(layers.Dense(1, activation='sigmoid'))
    model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model1.summary()

    # 2 hidden layers
    model2 = models.Sequential()
    model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model2.add(layers.MaxPooling2D((2, 2)))
    model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model2.add(layers.MaxPooling2D((2, 2)))
    model2.add(layers.Flatten())
    model2.add(layers.Dense(1, activation='sigmoid'))
    model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model2.summary()

    # 3 hidden layers
    model3 = models.Sequential()
    model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model3.add(layers.MaxPooling2D((2, 2)))
    model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model3.add(layers.MaxPooling2D((2, 2)))
    model3.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model3.add(layers.MaxPooling2D((2, 2)))
    model3.add(layers.Flatten())
    model3.add(layers.Dense(1, activation='sigmoid'))
    model3.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model3.summary()

    # load cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = select_flying_objects(y_train)
    y_test = select_flying_objects(y_test)

    model1.fit(x_train, y_train, epochs=6, batch_size=64)
    model2.fit(x_train, y_train, epochs=6, batch_size=64)
    model3.fit(x_train, y_train, epochs=6, batch_size=64)

    test_loss1, test_acc1 = model1.evaluate(x_test, y_test)
    print(f'test_acc1 = {test_acc1}')

    test_loss2, test_acc2 = model2.evaluate(x_test, y_test)
    print(f'test_acc2 = {test_acc2}')

    test_loss3, test_acc3 = model3.evaluate(x_test, y_test)
    print(f'test_acc3 = {test_acc3}')
