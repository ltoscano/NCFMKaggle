import tensorflow as tf 
import argparse 
import os 
import json
import numpy as np 

from misc.DataLoader import DataLoader 

slim = tf.contrib.slim

from nets import nets_factory
from preprocessing import preprocessing_factory as prepro
from datasets import dataloader 

from tensorflow.contrib.slim.python.slim.learning import train_step


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

def create_input_pipeline(dataset, prepro_fn, im_size, batch_size, shuffle=True):
    with tf.device('/cpu:0'):
        # setup training input pipeline 
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
          common_queue_capacity=20 * batch_size,
          common_queue_min=10 * batch_size,
          shuffle=shuffle)
        [image, label] = provider.get(['image', 'label'])

        image = prepro_fn(image, im_size, im_size)

        if shuffle:
            images, labels = tf.train.shuffle_batch(
              [image, label],
              batch_size=batch_size,
              num_threads=4,
              capacity=10*batch_size,
              min_after_dequeue=5*batch_size)
        else:
            images, labels = tf.train.batch(
              [image, label],
              batch_size=batch_size,
              num_threads=4,
              capacity=10*batch_size)

    batch_queue = slim.prefetch_queue.prefetch_queue([images, labels])
    images, labels = batch_queue.dequeue()
    return images, labels

def train_net(args):

    global_step = slim.create_global_step()
    im_size = 224 

    # get dataset, setup batch queue
    train_set = dataloader.get_dataset('train', args.data_dir) 
    train_preprocessing_fn = prepro.get_preprocessing('vgg_16', is_training=True)
    train_images, train_labels = create_input_pipeline(train_set, train_preprocessing_fn, im_size, args.batch_size)

    val_set = dataloader.get_dataset('val', args.data_dir)
    val_preprocessing_fn = prepro.get_preprocessing('vgg_16', is_training=False)
    val_images, val_labels = create_input_pipeline(val_set, val_preprocessing_fn, im_size, args.batch_size, shuffle=False)

    with tf.variable_scope('') as scope:
        logits, end_points = model(train_images)
        scope.reuse_variables()
        val_logits, val_end_points = model(val_images, is_training=False)

    # create networks
    # network_fn = nets_factory.get_network_fn('vgg_16', num_classes=8, weight_decay=.0005, is_training=True, finetune=args.finetune)
    # network_fn_val = nets_factory.get_network_fn('vgg_16', num_classes=8, is_training=False, finetune=False)

    # with tf.variable_scope('') as scope:
    #     logits, end_points = network_fn(train_images)
    #     scope.reuse_variables()
    #     val_logits, val_end_points = network_fn_val(val_images)

    val_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(val_logits, val_labels))

    # create loss and optimizer 
    cross_entropy_loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits, train_labels)
    regularization_loss = tf.add_n(tf.contrib.losses.get_regularization_losses())
    total_loss = tf.contrib.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(3e-4)

    # train op -- computes loss and updates gradients 
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # gather summaries 
    fw = tf.summary.FileWriter(args.save_folder, flush_secs=5)
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries.add(tf.summary.scalar('cross_entropy_loss', cross_entropy_loss))
    summaries.add(tf.summary.scalar('regularization_loss', regularization_loss))
    summaries.add(tf.summary.scalar('train_total_loss', total_loss))
    train_summary_op = tf.summary.merge(list(summaries))

    # total val_loss is evaluated at training time 
    total_val_loss = tf.placeholder(tf.float32)
    val_loss_summary_op = tf.summary.scalar('val_total_loss', total_val_loss)

    # initialize variables from start file 
    # restore_vars = slim.get_variables_to_restore(exclude=["vgg_16/fc6", "vgg_16/fc7"])
    # init_fn = slim.assign_from_checkpoint_fn(args.start_from, restore_vars, ignore_missing_vars=True)

    train_step_kwargs = {'best_val_loss': 0, 'save_folder': args.save_folder, 'num_val_samples': val_set.num_samples,
                         'batch_size': args.batch_size}
    saver = tf.train.Saver()

    def train_step_fn(session, *args, **kwargs):
        total_loss, should_stop = train_step(session, *args, **kwargs)

        train_dict = args[2]

        print('step {:.2f} - loss: {:.2f}'.format(train_step_fn.step, total_loss))
        train_summary = session.run(train_summary_op)
        fw.add_summary(train_summary, global_step=train_step_fn.step)

        if train_step_fn.step % 25 == 0:
            val_loss_sum = 0
            num_eval = 0
            while num_eval*train_dict['batch_size'] <= train_dict['num_val_samples']:
                batch_val_loss = session.run(val_loss)
                num_eval += 1
                val_loss_sum += batch_val_loss
            val_loss_run = val_loss_sum / float(num_eval)
            print('validation loss: {:.2f}'.format(val_loss_run))

            val_summary = session.run(val_loss_summary_op, feed_dict={total_val_loss: val_loss_run})
            fw.add_summary(val_summary, global_step=train_step_fn.step)

            if train_step_fn.step == 0:
                train_dict['best_val_loss'] = val_loss_run
            else:
                if val_loss_run < train_dict['best_val_loss']:
                    print('Saving model...')
                    train_dict['best_val_loss'] = val_loss_run
                    saver.save(session, os.path.join(train_dict['save_folder'], 'model.ckpt'))

        train_step_fn.step += 1

        return [total_loss, should_stop]

    train_step_fn.step = 0

    # disable summary and model saving 
    slim.learning.train(train_op, args.save_folder, train_step_kwargs=train_step_kwargs,
     train_step_fn=train_step_fn, summary_writer=None, save_summaries_secs=0, save_interval_secs=0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train a model')

    parser.add_argument('-data_dir', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/data/train', dest='data_dir')
    parser.add_argument('-start_from', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/vgg_16.ckpt', dest='start_from')

    parser.add_argument('-batch_size', type=int, default=64, dest='batch_size')
    parser.add_argument('-lr', type=float, default=3e-4, dest='lr')
    parser.add_argument('-finetune', action='store_true', help='finetune cnn')

    parser.add_argument('-save_name', type=str, default='linear-r1', dest='save_name')
    parser.add_argument('-save_folder', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/checkpoints', dest='save_folder')
    parser.add_argument('-save_every', type=int, default=50, dest='save_every')

    args = parser.parse_args()
    train_net(args)