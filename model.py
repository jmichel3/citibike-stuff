import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
# import tensorflow_addons as tfa

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        self.data = data
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        self.input_data = tf.placeholder(
            dtype=tf.float32, shape=[
                None, 3], name='data_in')
        self.target_data = tf.placeholder(
            dtype=tf.float32, shape=[
                None, 3], name='targets')

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(2))
        model.add(tf.keras.layers.Activation('softmax'))

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




