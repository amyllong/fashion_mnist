from icecream import ic

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
#import mnist_reader
import os # processing file path
import gzip # unzip the .gz file, not used here
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    
    with open(labels_path,'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    
    with open(images_path,'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


train_images, train_labels = load_mnist('./dataset/MNIST', kind='train')
test_images, test_labels = load_mnist('./dataset/MNIST', kind='t10k')
m_train = train_images.shape[0]
m_test = test_images.shape[0]


ic(train_images.shape)
ic(test_images.shape)

train_images, test_images = train_images.reshape([-1, 28, 28, 1]), test_images.reshape(-1, 28, 28, 1)
ic(train_images.shape)
ic(test_images.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

# Train_dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))\
           .shuffle(buffer_size=len(train_images))\
           .batch(batch_size=64)\
           .prefetch(buffer_size=64)\
           .repeat()

# Test dataset
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))\
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

model.fit(train_ds, batch_size = 64, epochs=5, steps_per_epoch=len(train_images)/64)

test_loss, test_acc = model.evaluate(test_ds, verbose = 2, steps=len(test_images)/64)
print('\nTest loss: ', test_loss)
print('Test accuracy:', test_acc)



try:
    del train_images, train_labels
    del test_images, test_labels
    print('Clear previously loaded data.')
except:
    pass