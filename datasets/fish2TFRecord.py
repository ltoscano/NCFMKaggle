from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf
from datasets import utils

_SPLIT_NAMES = {
    'train': ['train', 'val'],
    'test': ['test']
}

# The number of images in the validation set.
_NUM_FOLD = 5

# Seed for repeatability.
_RANDOM_SEED = 42

# The number of shards per dataset split.
_NUM_SHARDS = 5

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
      # Initializes function that decodes RGB JPEG data.
      self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
      self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
      image = self.decode_jpeg(sess, image_data)
      return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
      image = sess.run(self._decode_jpeg,
                       feed_dict={self._decode_jpeg_data: image_data})
      assert len(image.shape) == 3
      assert image.shape[2] == 3
      return image


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.
    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.
    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    directories = []
    class_names = []
    for filename in os.listdir(dataset_dir):
      path = os.path.join(dataset_dir, filename)
      if os.path.isdir(path):
        directories.append(path)
        class_names.append(filename)

    photo_filenames = []
    for directory in directories:
      for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'fishy_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, dataset_dir, save_dir, class2id=None):
    """Converts the given filenames to a TFRecord dataset.
    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
      image_reader = ImageReader()

      with tf.Session('') as sess:

        for shard_id in range(_NUM_SHARDS):
          output_filename = _get_dataset_filename(save_dir, split_name, shard_id)

          with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
            for i in range(start_ndx, end_ndx):
              sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                  i+1, len(filenames), shard_id))
              sys.stdout.flush()

              # Read the filename:
              image_data = tf.gfile.FastGFile(filenames[i], 'r').read()

              # class_id is None if no class2id mapping is passed (processing test set)
              class_id = class2id[class_from_filename(filenames[i])] if class2id else None

              image_id = os.path.basename(filenames[i])

              example = image_to_tfexample(image_data, 'jpg', class_id=class_id, image_id=image_id)
              tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

def class_from_filename(filename):
      return os.path.basename(os.path.dirname(filename))

def _dataset_exists(dataset_dir, split_names):
    for split_name in split_names:
      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)
        if not tf.gfile.Exists(output_filename):
          return False
    return True

def print_fold_stats(fold_filename_list, class2label):
    num_classes = len(class2label.keys())

    import numpy as np 
    for fold in range(_NUM_FOLD):
        print('FOLD %d -- %d images.' % (fold, len(fold_filename_list[fold])))
        dist = np.zeros(num_classes)
        for filename in fold_filename_list[fold]:
            dist[class2label[class_from_filename(filename)]] += 1
        print(dist/len(fold_filename_list[fold]))

def run(dataset_dir, save_dir, dataset='train'):
    """Runs the download and conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      dataset: train or test dataset
    """

    if not tf.gfile.Exists(dataset_dir):
      tf.gfile.MakeDirs(dataset_dir)

    # split_names = _SPLIT_NAMES[dataset]

    # if _dataset_exists(dataset_dir, split_names):
    #   print('Dataset files already exist. Exiting without re-creating them.')
    #   return

    if dataset == 'train':
        photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
        class2label = dict(zip(class_names, range(len(class_names))))

        # Divide into train and test:
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)

        file_name_folds = split(photo_filenames, _NUM_FOLD)
        print_fold_stats(file_name_folds, class2label)

        # convert the datasets 
        for fold in range(_NUM_FOLD):
            _convert_dataset('fold%d' % (fold), file_name_folds[fold], dataset_dir, save_dir, class2label)
        # _convert_dataset('train', train_filenames, dataset_dir, class2label)
        # _convert_dataset('val', val_filenames, dataset_dir, class2label)

        # Finally, write the labels file:
        label2class = dict(zip(range(len(class_names)), class_names))
        utils.write_label_file(label2class, save_dir)
        
    else:
        test_filenames = []
        for filename in os.listdir(dataset_dir):
            test_filenames.append(os.path.join(dataset_dir, filename))

        print('{} TEST'.format(len(test_filenames)))
        _convert_dataset('test', test_filenames, dataset_dir)

    # dataset_name = os.path.basename(dataset_dir)
    # print('Finished converting {} dataset to TFRecord {} split(s).\n'.format(dataset_name, '/'.join(split_names)))

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n)]

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, class_id=None, image_id=None):
    feature_dict = {
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format)
    }

    if class_id:
        feature_dict['image/class/label'] = int64_feature(class_id)

    if image_id:
        feature_dict['image/id'] = bytes_feature(image_id)

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

if __name__ == '__main__':
    data_dir = '/scratch/cluster/vsub/ssayed/NCFMKaggle/data/train/raw/'
    save_dir = '/scratch/cluster/vsub/ssayed/NCFMKaggle/data/train/cv/'
    run(data_dir, save_dir, dataset='train')