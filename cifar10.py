import tensorflow as tf
import numpy as np
import os
import cv2

from icecream import ic

from tensorflow.core.framework.types_pb2 import DataType
import cifar_tfrecord

record_file1 = 'cifar10Train.tfrecords'
dataset1 = tf.data.TFRecordDataset(record_file1)
dataType = cifar_tfrecord.getDataType()

record_file2 = 'cifar10Test.tfrecords'
dataset2 = tf.data.TFRecordDataset(record_file2)

#parse the tfrecord rile
def parse_record(record):
    name_to_features = {
        'dimension': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, name_to_features)

#decode the tfrecord file
def decode_record(record):
    image = tf.io.decode_raw(
        record['image_raw'], out_type=dataType, little_endian=True, fixed_length=None, name=None
    )
    label = record['label']
    dimension = record['dimension']
    image = tf.reshape(image, (dimension, dimension, 3))
    image = tf.image.random_flip_left_right(image, seed=None)
    image = tf.cast(image, tf.float32) / 255.0
    return (image, label)

#combine parse and decode 
def parse_and_decode(record):
    parsed_record = parse_record(record)
    decoded_record = decode_record(parsed_record)
    return decoded_record

#create training dataset
train_ds = dataset1.map(parse_and_decode)\
           .shuffle(buffer_size=10000)\
           .batch(batch_size=64)\
           .prefetch(buffer_size=64)\
           .repeat()

#create test dataset
test_ds = dataset2.map(parse_and_decode)\
            .batch(batch_size=64)\
            .prefetch(buffer_size=64)\
            .repeat()

#create model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train the model
model.fit(train_ds, batch_size = 64, epochs=60, steps_per_epoch=938)

#evaluate the model, aka test it
test_loss, test_acc = model.evaluate(test_ds, verbose = 2, steps=157)
print('\nTest loss: ', test_loss)
print('Test accuracy:', test_acc)