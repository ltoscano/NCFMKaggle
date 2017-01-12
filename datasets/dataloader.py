from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import utils

slim = tf.contrib.slim

_FILE_PATTERN = 'fishy_%s_*.tfrecord'

_SETSPLIT_SIZES = {
    'fold0': 756,
    'fold1': 756,
    'fold2': 755,
    'fold3': 755,
    'fold4': 755,
    'test': 1000
}

_NUM_CLASSES = 8

_NUM_SHARDS = 5
_NUM_FOLD = 5

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}

_LABELS_FILENAME = 'label2class.txt'

def get_dataset(split_name, fold_num, dataset_dir):

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
    if split_name == 'train':
        train_fold_names = ['fold%d' % i for i in range(_NUM_FOLD) if i != fold_num]
        file_pattern = [os.path.join(dataset_dir, _FILE_PATTERN % fold_name) for fold_name in train_fold_names]
        num_samples = sum([_SETSPLIT_SIZES[key] for key in _SETSPLIT_SIZES.keys() if key in train_fold_names])
    elif split_name == 'val':
        fold_name = 'fold%d' % fold_num
        file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % fold_name) 
        num_samples = _SETSPLIT_SIZES[fold_name]
    else:
        file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
        num_samples = _SETSPLIT_SIZES[split_name]

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        **extra_dataset_args)

if __name__ == '__main__':
    data_dir = '/scratch/cluster/vsub/ssayed/NCFMKaggle/data/train/cv/'
    split_name = 'val'
    fold_num = 1
    get_dataset(split_name, fold_num, data_dir)