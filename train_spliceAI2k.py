from model import *
from utils import *

from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

import math
import numpy as np
import time

print('eagerly?', tf.executing_eagerly())

# TRAINING PARAMETERS
batch_size = 128
num_classes = 3
epochs = 1

class DataGenerator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        return np.array(batch_x), np.array(batch_y)

start_time = time.time()

# importing the data
transcripts = np.loadtxt('./data/transcripts_chr21', dtype='str', delimiter='\t', max_rows=1000)
labels = np.loadtxt('./data/labels_chr21', dtype='str', delimiter='\t', max_rows=1000)

# one-hot-encoding
transcripts, labels = transform_input(transcripts, labels)

transcripts = np.array(transcripts)
labels = np.array(labels)

(x_train, x_test, y_train, y_test) = train_test_split(transcripts,
                                                      labels, test_size=0.2)

input_shape = x_train.shape[1:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

print("Data prep: {} seconds".format(time.time() - start_time))

lr_scheduler = LearningRateScheduler(lr_schedule)

model = spliceAI_model(input_shape=input_shape)

model.compile(loss=custom_crossentropy_loss,
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])

print(model.summary())

start_time = time.time()
training_generator = DataGenerator(x_train, y_train, batch_size)

for e in range(1, 10):
    model.fit(training_generator, epochs=e+1, initial_epoch=e, callbacks=[lr_scheduler], shuffle=True)
    y_pred = model.predict(x_test)
    if topk_accuracy(y_test, y_pred)>0.70:
        break

print("Fitting: {} seconds".format(time.time() - start_time))

model.save('./data/model_spliceAI2k')

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

y_pred = model.predict(x_test)

y_test, y_pred = transform_output(y_test, y_pred)

donor_t_p, acceptor_t_p, blank_t_p = 0, 0, 0
donor, acceptor, blank = 0, 0, 0

for i in range(len(y_test)):
    for j in range(len(y_test[0])):
        if y_test[i][j] == y_pred[i][j] and y_test[i][j] == 'd':
            donor += 1
            donor_t_p += 1
        elif y_test[i][j] == y_pred[i][j] and y_test[i][j] == 'a':
            acceptor += 1
            acceptor_t_p += 1
        elif y_test[i][j] == y_pred[i][j] and y_test[i][j] == 'b':
            blank += 1
            blank_t_p += 1
        elif y_test[i][j] == 'd':
            donor += 1
        elif y_test[i][j] == 'a':
            acceptor += 1
        elif y_test[i][j] == 'b':
            blank += 1

print(
    "Out of {} blank {} TP, out of {} donor {} TP, out of {} acceptor {} TP".format(blank, blank_t_p, donor, donor_t_p,
                                                                                    acceptor, acceptor_t_p))
