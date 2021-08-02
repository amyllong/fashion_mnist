'''Used local data and changed input pipeline'''

from icecream import ic
import datetime

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

ic(m_train)
ic(m_test)

print('\n')
ic(train_images.shape[1])
ic(test_images.shape[1])

print('\n')
ic(train_images.shape)
ic(test_images.shape)

print('\nReshaped to -1, 28, 28, 1')
train_images, test_images = train_images.reshape([-1, 28, 28, 1]), test_images.reshape(-1, 28, 28, 1)
ic(train_images.shape)
ic(test_images.shape)
ic(train_images.shape[1])
ic(train_images.shape[1])

print()
ic(train_images.dtype)
ic(test_images.dtype)

train_images = train_images / 255.0
test_images = test_images / 255.0

ic(train_images.dtype)
ic(test_images.dtype)

indexes = np.arange(train_images.shape[0])
for _ in range(5): indexes = np.random.permutation(indexes) #shuffle 5 times, not technically necessary
train_images = train_images[indexes]
train_labels = train_labels[indexes]

val_count = 48000
val_images = train_images[val_count:]
val_labels = train_labels[val_count:]
train_images = train_images[:val_count]
train_labels = train_labels[:val_count]

ic(train_images.shape)
ic(val_images.shape)
ic(test_images.shape)

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

#checkpoint added 
mypath = './'
model_path = 'model.{epoch:02d}-{val_loss:.2f}.h5'
log_dir="./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_checkpoint_callback = [
    tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, period=5),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir)
]

model.fit(train_ds, batch_size = 64, epochs=10, steps_per_epoch=len(train_images)/64, validation_data=(val_images, val_labels), callbacks=[model_checkpoint_callback])

test_loss, test_acc = model.evaluate(test_ds, verbose = 2, steps=len(test_images)/64)
print('\nTest loss: ', test_loss)
print('Test accuracy:', test_acc)

model.save(model_path)

try:
    del train_images, train_labels
    del test_images, test_labels
    print('Clear previously loaded data.')
except:
    pass