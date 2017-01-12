import tensorflow as tf 
import argparse 
import os 
import json
import numpy as np 

slim = tf.contrib.slim

from preprocessing import preprocessing_factory as prepro
from datasets import dataloader 

from tensorflow.contrib.slim.python.slim.learning import train_step
import time 
from distutils import dir_util

def model(inputs, num_classes=8, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True, weight_decay=.0005):
  with tf.variable_scope('model') as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d],
                        activation_fn = tf.nn.relu,
                        biases_initializer=tf.zeros_initializer,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        outputs_collections=end_points_collection):
        net = slim.conv2d(inputs, 8, [9, 9], stride=2, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 32, [5, 5], scope='conv2') 
        net = slim.max_pool2d(net, [2, 2], scope='pool2') 
        net = slim.conv2d(net, 64, [3, 3], scope='conv3') 
        net = slim.max_pool2d(net, [2, 2], scope='pool3') 
        net = slim.conv2d(net, 128, [3, 3], scope='conv4') 
        net = slim.max_pool2d(net, [2, 2], scope='pool4') 
        net = slim.conv2d(net, 128, [3, 3], scope='conv5') 
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        with slim.arg_scope([slim.conv2d], weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
            net = slim.conv2d(net, 256, [3, 3], padding='VALID', scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 256, [1, 1], scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
            net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            end_points[sc.name + '/fc8'] = net
        return net, end_points

def create_input_pipeline(dataset, prepro_fn, batch_size, shuffle=True):
    im_size = 224

    with tf.device('/cpu:0'):
        # setup training input pipeline 
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
          common_queue_capacity=20 * batch_size,
          common_queue_min=10 * batch_size,
          num_epochs=1,
          shuffle=shuffle,
          num_readers=30)
        [image, label] = provider.get(['image', 'label'])

        image = prepro_fn(image, im_size, im_size)

        if shuffle:
            images, labels = tf.train.shuffle_batch(
              [image, label],
              batch_size=batch_size,
              num_threads=10,
              capacity=10*batch_size,
              min_after_dequeue=5*batch_size,
              allow_smaller_final_batch=True)
        else:
            images, labels = tf.train.batch(
              [image, label],
              batch_size=batch_size,
              num_threads=4,
              capacity=10*batch_size)

    # batch_queue = slim.prefetch_queue.prefetch_queue([images, labels])
    # images, labels = batch_queue.dequeue()
    return images, labels

def create_train_val_graphs(args, fold_num):

    global_step = slim.get_or_create_global_step()

    train_set = dataloader.get_dataset('train', fold_num, args.data_dir) 
    val_set = dataloader.get_dataset('val', fold_num, args.data_dir)

    train_preprocessing_fn = prepro.get_preprocessing('vgg_16', is_training=True)
    val_preprocessing_fn = prepro.get_preprocessing('vgg_16', is_training=False)

    train_images, train_labels = create_input_pipeline(train_set, train_preprocessing_fn, args.batch_size)
    val_images, val_labels = create_input_pipeline(val_set, val_preprocessing_fn, args.batch_size, shuffle=False)

    with tf.variable_scope('') as scope:
        logits, end_points = model(train_images)
        scope.reuse_variables()
        val_logits, val_end_points = model(val_images, is_training=False)

    val_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(val_logits, val_labels))

    # create loss and optimizer 
    cross_entropy_loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits, train_labels)
    regularization_loss = tf.add_n(tf.contrib.losses.get_regularization_losses())
    total_loss = tf.contrib.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(3e-4)

    # train op -- computes loss and updates gradients 
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    return train_op, val_loss, train_set.num_samples, val_set.num_samples

def train_net(args):

    _NUM_FOLDS = 5

    print('saving initial models...')
    # create and save initial models
    train_op, val_loss, num_train_samples, num_val_samples = create_train_val_graphs(args, 0)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        for i in range(_NUM_FOLDS):
            saver.save(sess, os.path.join(args.save_folder, 'fold%d/cur/model' % i))

    saver = tf.train.Saver()
    def train_step_fn(session, *args, **kwargs):
        total_loss, should_stop = train_step(session, *args, **kwargs)

        train_args = args[2]
        train_args['step'] += 1

        step = train_args['step']

        # print('step {:.2f} - loss: {:.2f}'.format(step, total_loss))

        train_args['train_loss_sum'] += total_loss 

        num_train = train_args['num_train']
        batch_size = train_args['batch_size']

        if step == np.ceil(num_train/float(batch_size)):
            should_stop = True
            val_loss_op = train_args['val_loss_op']
            val_loss_sum = 0
            num_eval = 0

            while num_eval < np.ceil(train_args['num_val']/batch_size):
                val_loss_sum += session.run(val_loss_op)
                num_eval += 1
            train_args['val_loss'] = val_loss_sum / float(num_eval)
            train_args['train_loss'] = train_args['train_loss_sum']/step
            fold_num = train_args['fold_num']
            print('fold {} -- training loss: {:.2f}'.format(fold_num, train_args['train_loss']))
            print('fold {} -- validation loss: {:.2f}'.format(fold_num, train_args['val_loss']))

            save_running_model = os.path.join(train_args['save_folder'], 'fold%d/cur/model' % train_args['fold_num'])
            saver.save(session, save_running_model)

        return [total_loss, should_stop]

    best_cv_loss = float('inf')
    epoch_num = 0
    while True:
        print('epoch {}'.format(epoch_num))
        cv_loss_sum = 0
        for fold in range(_NUM_FOLDS):
            tf.reset_default_graph()
            train_op, val_loss, num_train_samples, num_val_samples = create_train_val_graphs(args, fold)

            init_fn = slim.assign_from_checkpoint_fn(os.path.join(args.save_folder, 'fold%d/cur/model' % fold), slim.get_variables_to_restore())
            train_step_kwargs = {'step': 0, 'val_loss_op': val_loss, 'batch_size': args.batch_size, 'num_train': num_train_samples, 
                                'num_val': num_val_samples, 'save_folder': args.save_folder, 'fold_num': fold, 'train_loss_sum': 0}
            slim.learning.train(train_op, logdir=None, init_fn=init_fn, train_step_kwargs=train_step_kwargs, train_step_fn=train_step_fn)
            cv_loss_sum += train_step_kwargs['val_loss']
        cv_loss = cv_loss_sum / float(_NUM_FOLDS)
        print('cross validation loss: {:.2f}'.format(cv_loss))
        if cv_loss < best_cv_loss:
            print('saving models...')
            best_cv_loss = cv_loss
            for fold in range(_NUM_FOLDS):
                cur_dir = os.path.join(args.save_folder, 'fold%d/cur/' % fold)
                best_dir = os.path.join(args.save_folder, 'fold%d/best/' % fold)
                dir_util.copy_tree(cur_dir, best_dir)
        epoch_num += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train a model')

    parser.add_argument('-data_dir', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/data/train/cv', dest='data_dir')
    parser.add_argument('-start_from', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/vgg_16.ckpt', dest='start_from')

    parser.add_argument('-batch_size', type=int, default=64, dest='batch_size')
    parser.add_argument('-lr', type=float, default=3e-4, dest='lr')
    parser.add_argument('-finetune', action='store_true', help='finetune cnn')

    parser.add_argument('-save_name', type=str, default='linear-r1', dest='save_name')
    parser.add_argument('-save_folder', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/checkpoints/cv', dest='save_folder')
    parser.add_argument('-save_every', type=int, default=50, dest='save_every')

    args = parser.parse_args()
    train_net(args)
