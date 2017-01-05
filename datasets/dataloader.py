from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import utils

slim = tf.contrib.slim

_FILE_PATTERN = 'fishy_%s_*.tfrecord'

_SETSPLIT_SIZES = {
    'train/train': 3022,
    'train/val': 755,
    'test_stg1/test': 1000
}

_NUM_CLASSES = 8

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}

_LABELS_FILENAME = 'label2class.txt'

def get_dataset(split_name, dataset_dir):

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/id': tf.FixedLenFeature((), tf.string, default_value='')
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'id': slim.tfexample_decoder.Tensor('image/id')
    }

    extra_dataset_args = {}
    if split_name != 'test':
        keys_to_features['image/class/label'] = tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        items_to_handlers['label'] = slim.tfexample_decoder.Tensor('image/class/label')
        labels_to_names = utils.read_label_file(dataset_dir)
        extra_dataset_args['label2class'] = labels_to_names

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
    dataset_name = os.path.basename(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=_SETSPLIT_SIZES['{}/{}'.format(dataset_name, split_name)],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        **extra_dataset_args)