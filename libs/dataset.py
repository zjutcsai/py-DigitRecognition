# -*- coding: utf-8 -*-
import os
import gzip
import numpy as np
try:
    from urllib import urlretrieve 
except:
    from urllib.request import urlretrieve
    
"""Code from Tensorflow's mnist.py"""
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

  
def extract_labels(f, one_hot=False, num_classes=10):
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number {} in MNIST label file: {}'.firmat(magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels

def extract_images(f):
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number {} in MNIST image file: {}'.format(magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data
    
def download_mnist(data_dir):
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    # create directory if path is not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # download train images and extract
    if not os.path.exists(os.path.join(data_dir, TRAIN_IMAGES)):
        url = os.path.join(SOURCE_URL, TRAIN_IMAGES)
        urlretrieve(url, os.path.join(data_dir, TRAIN_IMAGES))
        print('Successfully download {}'.format(TRAIN_IMAGES))
    with open(os.path.join(data_dir, TRAIN_IMAGES), 'rb') as f:
        train_images = extract_images(f)
    
    # download train labels and extract
    if not os.path.exists(os.path.join(data_dir, TRAIN_LABELS)):
        url = os.path.join(SOURCE_URL, TRAIN_LABELS)
        urlretrieve(url, os.path.join(data_dir, TRAIN_LABELS))
        print('Successfully download {}'.format(TRAIN_LABELS))
    with open(os.path.join(data_dir, TRAIN_LABELS), 'rb') as f:
        train_labels = extract_labels(f, one_hot=True)
    
    # download test images and extract
    if not os.path.exists(os.path.join(data_dir, TEST_IMAGES)):
        url = os.path.join(SOURCE_URL, TEST_IMAGES)
        urlretrieve(url, os.path.join(data_dir, TEST_IMAGES))
        print('Successfully download {}'.format(TEST_IMAGES))
    with open(os.path.join(data_dir, TEST_IMAGES), 'rb') as f:
        test_images = extract_images(f)
    
    # download test labels and extract
    if not os.path.exists(os.path.join(data_dir, TEST_LABELS)):
        url = os.path.join(SOURCE_URL, TEST_LABELS)
        urlretrieve(url, os.path.join(data_dir, TEST_LABELS))
        print('Successfully download {}'.format(TEST_LABELS))
    with open(os.path.join(data_dir, TEST_LABELS), 'rb') as f:
        test_labels = extract_labels(f, one_hot=True)
    
    print('Training data info:', train_images.shape, train_labels.shape)
    print('Testing  data info:', test_images.shape, test_labels.shape)
    
    return train_images, train_labels, test_images, test_labels