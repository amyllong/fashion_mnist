'''Used local data'''

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


train_images, train_labels = load_mnist('./dataset', kind='train')
test_images, test_labels = load_mnist('./dataset', kind='t10k')
m_train = train_images.shape[0]
m_test = test_images.shape[0]


ic(train_images.shape)
ic(test_images.shape)

train_images, test_images = train_images.reshape([-1, 28, 28, 1]), test_images.reshape(-1, 28, 28, 1)
ic(train_images.shape)
ic(test_images.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

#train_images, test_images = train_images.reshape(-1,28,28,1), test_images.reshape(-1,28,28,1)
model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.8))
'''
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))'''
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.8))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size = 64, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest accuracy:', test_acc)



try:
    del train_images, train_labels
    del test_images, test_labels
    print('Clear previously loaded data.')
except:
    pass