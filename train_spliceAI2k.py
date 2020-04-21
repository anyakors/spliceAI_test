from __future__ import print_function
import keras
from keras.layers import Dense, Conv1D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D, Cropping1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, Sequential
from keras.utils import to_categorical, plot_model
from keras.datasets import cifar10
from keras import optimizers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import os


def hot_encode_seq(let):
    #hot-encode the sequence, where "O" corresponds to zero-padded areas
    if let=='A':
        return([1,0,0,0])
    elif let=='T':
        return([0,1,0,0])
    elif let=='C':
        return([0,0,1,0])
    elif let=='G':
        return([0,0,0,1])
    elif let=='O':
        return([0,0,0,0])

def hot_encode_label(let):
    #hot-encode the labels, where "p" corresponds to zero-padded areas
    if let=='p':
        return([0,0,0])
    elif let=='b':
        return([1,0,0])
    elif let=='a':
        return([0,1,0])
    elif let=='d':
        return([0,0,1])


def lr_schedule(epoch):
    #learning rate scheduler
    lr = 0.001
    if epoch == 7:
        lr *= 0.5
    elif epoch == 8:
        lr *= (0.5)**2
    elif epoch == 9:
        lr *= (0.5)**3
    elif epoch == 10:
        lr *= (0.5)**4
    print('Learning rate: ', lr)
    return lr

# TRAINING PARAMETERS
batch_size = 12
num_classes = 3
epochs = 10
data_augmentation = False

def RB_block(inputs,
             num_filters=32,
             kernel_size=11,
             strides=1,
             activation='relu',
             dilation_rate=1):
    """1D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        dilation rate (int): dilation rate

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  dilation_rate=dilation_rate)

    x = inputs

    for layer in range(2):
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = conv(x)

    return x


def spliceAI_model(input_shape, num_classes=3):
    """Model builder

    Shortcut layers after every 4 RB blocks.

    # Arguments
        input_shape (tensor): shape of input image tensor
        num_classes (int): number of classes

    # Returns
        model (Model): Keras model instance
    """

    inputs = Input(shape=input_shape)

    # initiate 
    x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(inputs)

    # shortcut 1: just another Conv on x
    y = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)

    # RB 1: 32 11 1
    for stack in range(4):
        x = RB_block(x, num_filters=32, kernel_size=11, strides=1, activation='relu', dilation_rate=1)

    # shortcut 2: Conv on x + add to existing shortcut
    y = keras.layers.add([Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x), y])

    # RB 2: 32 11 4
    for stack in range(4):
        x = RB_block(x, num_filters=32, kernel_size=11, strides=1, activation='relu', dilation_rate=4)

    # shortcut 3: Conv on x + add to existing shortcut
    y = keras.layers.add([Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x), y])  

    # RB 3: 32 21 10
    for stack in range(4):
        x = RB_block(x, num_filters=32, kernel_size=21, strides=1, activation='relu', dilation_rate=10)

    # another Conv on x
    x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)

    # now adding up with what was shortcut from the prev layers
    x = keras.layers.add([x, y]) 

    # final Conv
    x = Conv1D(3, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)

    x = Dense(num_classes, activation='softmax')(x)

    # crop to fit the labels (7k to 5k)
    outputs = Cropping1D(cropping=(1000, 1000))(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


transcripts_ = np.loadtxt('./data/transcripts', dtype='str', delimiter='\t')
labels_ = np.loadtxt('./data/labels', dtype='str', delimiter='\t')

transcripts = []
labels = []

# hot-encode
for i in range(len(transcripts_)):
    # hot-encode seq
    transcripts.append([np.array(hot_encode_seq(let)) for let in transcripts_[i]])
    # hot-encode labels
    labels.append([np.array(hot_encode_label(x)) for x in labels_[i]])

transcripts = np.array(transcripts)
labels = np.array(labels)

(x_train, x_test, y_train, y_test) = train_test_split(transcripts,
    labels, test_size=0.2)

input_shape = x_train.shape[1:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

lr_scheduler = LearningRateScheduler(lr_schedule)

model = spliceAI_model(input_shape=input_shape)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[lr_scheduler], validation_data=(x_test, y_test), shuffle=True)

model.save('./data/model_spliceAI2k')

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

predictions = model.predict(x_test)

print(predictions)

print(classification_report(y_test.argmax(axis=1),
    predictions.argmax(axis=1), target_names=['Blank', 'Donor', 'Acceptor']))

print(confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1)))