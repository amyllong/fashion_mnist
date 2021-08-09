from icecream import ic
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
from tensorflow.core.framework.types_pb2 import DataType

from mpl_toolkits.axes_grid1 import ImageGrid

import convert_tfrecord2

record_file = 'fmnistTrain.tfrecords'
dataset = tf.data.TFRecordDataset(record_file, buffer_size=100)
dataType = convert_tfrecord2.getDataType()

record_file2 = 'fmnistTest.tfrecords'
dataset2 = tf.data.TFRecordDataset(record_file2, buffer_size=100)

def parse_record(record):
    name_to_features = {
        'dimension': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, name_to_features)

def decode_record(record):
    image = tf.io.decode_raw(
        record['image_raw'], out_type=dataType, little_endian=True, fixed_length=None, name=None
    )
    label = record['label']
    dimension = record['dimension']
    image = tf.reshape(image, (dimension, dimension, 1))
    image = tf.image.random_flip_up_down(image, seed=None)
    image = tf.cast(image, tf.float32) / 255.0
    return (image, label)

def parse_and_decode(record):
    parsed_record = parse_record(record)
    decoded_record = decode_record(parsed_record)
    return decoded_record

train_ds = tf.data.TFRecordDataset(record_file).map(parse_and_decode)\
           .shuffle(buffer_size=10000)\
           .batch(batch_size=64)\
           .prefetch(buffer_size=64)\
           .repeat()

test_ds = tf.data.TFRecordDataset(record_file2).map(parse_and_decode)\
            .batch(batch_size=64)\
            .prefetch(buffer_size=64)\
            .repeat()


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, batch_size = 64, epochs=5, steps_per_epoch=938)


test_loss, test_acc = model.evaluate(test_ds, verbose = 2, steps=157)
print('\nTest accuracy:', test_acc)

try:
    del train_ds
    del test_ds
    print('Clear previously loaded data.')
except:
    pass
############################################## Check the images