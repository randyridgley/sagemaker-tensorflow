"""Converts MNIST data to TFRecords file format with Example protos."""
import os
import tensorflow as tf
import sys
import tarfile
from six.moves import urllib
from ipywidgets import FloatProgress
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name, directory):
    """Converts a dataset to tfrecords."""
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def cifar10_download(data_dir='/tmp/cifar10_data', print_progress=True):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(os.path.join(data_dir, 'cifar-10-batches-bin')):
        print('cifar dataset already downloaded')
        return

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        f = FloatProgress(min=0, max=100)
        display(f)
        sys.stdout.write('\r>> Downloading %s ' % (filename))        

        def _progress(count, block_size, total_size):
            if print_progress:
                f.value = 100.0 * count * block_size / total_size

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(data_dir)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
