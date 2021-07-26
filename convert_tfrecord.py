import tensorflow as tf
import os
import sys
import argparse

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label):
    image_shape = tf.image.decode_png(image_string).shape
    feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

##############################################################################
def _data_path(data_directory:str, name:str) -> str:
    '''Construct a full path to a TFRecord file to be stored in the 
    data_directory. Will also ensure the data directory exists
    
    Args:
        data_directory: The directory where the records will be stored
        name:           The name of the TFRecord
    
    Returns:
        The full path to the TFRecord file'''
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)
    return os.path.join(data_directory, f'{name}.tfrecords')

def convert_to(data_set, name:str, data_directory:str, num_shards:int=1):
    """Convert the dataset into TFRecords on disk
    
    Args:
        data_set:       The MNIST data set to convert
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
    """
    print(f'Processing {name} data')
    images = data_set.images
    labels = data_set.labels
    num_examples, rows, cols, depth = data_set.images.shape

def _process_examples(start_idx:int, end_index:int, filename:str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(start_idx, end_index):
            sys.stdout.write(f"\rProcessing sample {index+1} of {num_examples}")
            sys.stdout.flush()
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)
        }))
    writer.write(example.SerializeToString())

    if num_shards == 1:
        _process_examples(0, data_set.num_examples, _data_path(data_directory, name))
    else:
        total_examples = data_set.num_examples
        samples_per_shard = total_examples // num_shards
    for shard in range(num_shards):
        start_index = shard * samples_per_shard
        end_index = start_index + samples_per_shard
        _process_examples(start_index, end_index, _data_path(data_directory, f'{name}-{shard+1}'))
    print()