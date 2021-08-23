import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

folders = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
images = []
labels = []

def load_images(folder):
    images = []
    labels = []
    root_folder = './cifar10/test'
    #ic(os.path.join(root_folder, folder))
    count = 0
    for filename in os.listdir(os.path.join(root_folder, folder)):
        #if(count == 0):
            #ic(os.path.join(root_folder, folder, filename))
        #count = 1
        img = cv2.imread(os.path.join(root_folder, folder, filename))
        if img is not None:
            images.append(img)
            labels.append(int(folder))
    return images, labels

for folder in folders:
    temp_images, temp_labels = load_images(folder)
    images = images + temp_images
    labels = labels + temp_labels

##########
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create the features dictionary.
def image_example(image, label, dimension):
    feature = {
        'dimension': _int64_feature(dimension),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

record_file = 'cifar10Test.tfrecords'
dimension = 32

dataType = images[0].dtype
def getDataType():
    return dataType

#write the tfrecord file
with tf.io.TFRecordWriter(record_file) as writer:
   for i in range(len(images)):
      image = images[i]
      label = labels[i]
      tf_example = image_example(image, label, dimension)
      writer.write(tf_example.SerializeToString())

#read in TFRecord file
dataset = tf.data.TFRecordDataset(record_file, buffer_size=100)

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
    image = tf.reshape(image, (dimension, dimension, 3))
    return (image, label)

    
for record in dataset:
    parsed_record = parse_record(record)
    decoded_record = decode_record(parsed_record)
    image, label = decoded_record
    print(image.shape, label.shape)
    break

############################################## Check the images
im_list = []
n_samples_to_show = 16
c = 0
for record in dataset:
  c+=1
  if c > n_samples_to_show:
    break
  parsed_record = parse_record(record)
  decoded_record = decode_record(parsed_record)
  image, label = decoded_record
  im_list.append(image)


# Visualization
fig = plt.figure(figsize=(4., 4.))
# Ref: https://matplotlib.org/3.1.1/gallery/axes_grid1/simple_axesgrid.html
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
# Show image grid
for ax, im in zip(grid, im_list):
    # Iterating over the grid returns the Axes.
    ax.imshow(im, 'gray')
plt.show()

