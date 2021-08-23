from icecream import ic
from numpy import testing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

#from sklearn.model_selection import train_test_split

from tensorflow.core.framework.types_pb2 import DataType
import convert_tfrecord2

#get tfrecord files
record_file = 'fmnistTrain.tfrecords'
dataset = tf.data.TFRecordDataset(record_file)
dataType = convert_tfrecord2.getDataType()

record_file2 = 'fmnistTest.tfrecords'
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
    image = tf.reshape(image, (dimension, dimension, 1))
    image = tf.image.random_flip_left_right(image, seed=None)
    image = tf.cast(image, tf.float32) / 255.0
    return (image, label)

#combine parse and decode 
def parse_and_decode(record):
    parsed_record = parse_record(record)
    decoded_record = decode_record(parsed_record)
    return decoded_record

#create training dataset
train_ds = dataset.map(parse_and_decode)\
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
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#create checkpoints
model_path = 'model1.h5'
model_checkpoint_callback = [
    tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, period=5),
]

#train the model
#model.fit(train_ds, batch_size = 64, epochs=5, steps_per_epoch=938)

#evaluate the model, aka test it
#test_loss, test_acc = model.evaluate(test_ds, verbose = 2, steps=157)
#print('\nTest loss: ', test_loss)
#print('Test accuracy:', test_acc)

#save the model
#model.save(model_path)

def test_predict():
    images = []
    root_folder = './data/test/0'
    for filename in os.listdir(root_folder):
        for i in range(10):
            img = cv2.imread(os.path.join(root_folder, filename), 0)
            if img is not None:
                images.append(img)
        return images


images = test_predict()

for image in images:
    image.reshape(28, 28, 1)
    image = tf.cast(image, tf.float32) / 255.0

reconstructed_model = tf.keras.models.load_model(model_path)
print(model.predict(images).argmax(1))
print(reconstructed_model.predict(images).argmax(1))



#clear everything
try:
    del train_ds
    del test_ds
    print('Clear previously loaded data.')
except:
    pass
############################################## Check the images