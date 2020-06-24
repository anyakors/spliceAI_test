from utils import *
import numpy as np

import keras
from keras.layers import Dense, Conv1D, BatchNormalization, Activation
from keras.layers import Input, Cropping1D
from keras.models import Model

import tensorflow as tf

print('eagerly?', tf.executing_eagerly())


def lr_schedule(epoch):
    lr = 0.001
    if epoch == 7:
        lr *= 0.5
    elif epoch == 8:
        lr *= 0.5 ** 2
    elif epoch == 9:
        lr *= 0.5 ** 3
    elif epoch == 10:
        lr *= 0.5 ** 4
    print('Learning rate: ', lr)
    return lr


# @tf.function
def custom_crossentropy_loss(y_true, y_pred):
    # clip the predicted values so we never have to calc log of 0
    # norm the probas so the sum of all probas for one observation is 1
    y_pred /= tf.expand_dims(tf.reduce_sum(y_pred, axis=-1), -1)
    y_pred = tf.keras.backend.clip(y_pred, 1e-15, 1 - 1e-15)

    # mask for blank, donor, acceptor sites 
    mask_b = np.array([True, False, False])
    mask_a = np.array([False, True, False])
    mask_d = np.array([False, False, True])

    # normalize the labels by the number of samples of each class
    labels_b = tf.squeeze(tf.boolean_mask(y_true, mask_b, axis=2))
    labels_b /= tf.add(tf.expand_dims(tf.reduce_sum(labels_b, axis=-1), -1), tf.constant(1e-15))

    labels_a = tf.squeeze(tf.boolean_mask(y_true, mask_a, axis=2))
    labels_a /= tf.add(tf.expand_dims(tf.reduce_sum(labels_a, axis=-1), -1), tf.constant(1e-15))

    labels_d = tf.squeeze(tf.boolean_mask(y_true, mask_d, axis=2))
    labels_d /= tf.add(tf.expand_dims(tf.reduce_sum(labels_d, axis=-1), -1), tf.constant(1e-15))

    # stack everything back normalized
    labels_norm = tf.stack([labels_b, labels_a, labels_d], axis=-1)

    return -tf.reduce_sum(labels_norm * tf.keras.backend.log(y_pred))


def topk_accuracy_(y_true, y_pred):

    y_true = y_true.reshape([len(y_true)*5000, 3])
    y_pred = y_true.reshape([len(y_pred)*5000, 3])

    a_true, d_true = np.nonzero(y_true[:, 1]), np.nonzero(y_true[:, 2])
    k = len(a_true[0])

    a_pred, d_pred = y_pred[:, 1], y_pred[:, 2]
    a_pred_topk = np.argsort(a_pred, axis=-1)[-k:]
    d_pred_topk = np.argsort(d_pred, axis=-1)[-k:]

    accuracy = len(np.intersect1d(a_true, a_pred_topk)) / len(a_true) + \
               len(np.intersect1d(d_true, d_pred_topk)) / len(d_true)

    return accuracy


def RB_block(inputs,
             num_filters=32,
             kernel_size=11,
             strides=1,
             activation='relu',
             dilation_rate=1):
    """1D Convolution-Batch Normalization-Activation stack builder
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
    """
    inputs = Input(shape=input_shape)

    # initiate 
    x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(inputs)
    # another Conv on x before splitting
    y = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)

    d = [1, 4, 10] #dilation
    for i in range(3):
        # RB 1, 2, 3: 32 11 1, 4, 10
        for stack in range(4):
            x = RB_block(x, num_filters=32, kernel_size=11, strides=1, activation='relu', dilation_rate=d[i])
        if i==0 or i==1:
            y = keras.layers.add([Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x), y])

    x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)
    # adding up with what was shortcut from the prev layers
    x = keras.layers.add([x, y])
    x = Conv1D(3, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)
    x = Dense(num_classes, activation='softmax')(x)
    # crop to fit the labels (7k to 5k)
    outputs = Cropping1D(cropping=(1000, 1000))(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def topk_accuracy(y_test, y_pred):

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

    return (0.5*(donor_t_p/donor + acceptor_t_p/acceptor))