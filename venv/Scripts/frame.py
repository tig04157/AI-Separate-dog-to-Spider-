# from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy
import sys
import tensorflow as tf
# import dataset
import os
import cv2
from sklearn.model_selection import train_test_split
# from Scripts.dataset import dataset

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

path = '../.image/stotal/total/'
categori = ['ha', 'bu']
number = 50

X = []

for num in range(len(categori)):
    for i in range(number):
        img = cv2.imread(path + categori[num] + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
        # print(path + categori[num] + str(i) + '.jpg')
        X.append(img/255)
        # print(img)

Y = []
for i in range(100):
    if i<50:
        Y.append("0")
    else:
        Y.append("1")

X = numpy.array(X)
Y = numpy.array(Y)
X = numpy.reshape(X,(-1, 100, 100, 1))
Y = np_utils.to_categorical(Y)
# print(X[0][50])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=34)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 1)))

model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(59, activation='relu'))
model.add(Dense(12, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=200, verbose=1, callbacks=[checkpointer, early_stopping_callback])

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label = "Testset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label = "Trainset_loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
