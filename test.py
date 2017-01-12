import tensorflow as tf 
import argparse 
import os 
import csv 
import numpy as np 

slim = tf.contrib.slim

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

def create_input_pipeline(dataset, prepro_fn, batch_size):
    im_size = 224
    with tf.device('/cpu:0'):
        # setup training input pipeline 
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
          common_queue_capacity=20 * batch_size,
          common_queue_min=10 * batch_size,
          num_epochs=1)
        [image, image_id] = provider.get(['image', 'id'])

        image = prepro_fn(image, im_size, im_size)

        images, image_ids = tf.train.batch([image, image_id], batch_size=batch_size, num_threads=4, capacity=10*batch_size,
                                            allow_smaller_final_batch=True)
    return images, image_ids

def create_test_graph(args, fold_num):

    global_step = slim.get_or_create_global_step()

    test_set = dataloader.get_dataset('test', fold_num, args.data_dir) 

    prepro_fn = prepro.get_preprocessing('vgg_16', is_training=False)

    test_images, test_image_ids = create_input_pipeline(test_set, prepro_fn, args.batch_size)

    logits, end_points = model(test_images, is_training=False)
    softmax = tf.nn.softmax(logits)

    return softmax, test_image_ids, test_set.num_samples

def test_net(args):

    ids_to_probs = {}
    for fold in range(5):
        tf.reset_default_graph()

        softmax, test_image_ids, num_test_samples = create_test_graph(args, fold)

        restore_vars = slim.get_model_variables()
        load_file = os.path.join(args.start_from, 'fold%d/best/model' % fold)
        init_fn = slim.assign_from_checkpoint_fn(load_file, restore_vars, ignore_missing_vars=True)

        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(local_init_op)
            init_fn(sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            num_eval = 0
            test_probs = []
            test_ids = []
            while num_eval < num_test_samples:
                image_ids, probs = sess.run([test_image_ids, softmax])
                num_eval += len(probs)
                print('{}/{}'.format(num_eval, num_test_samples))
                test_probs += list(probs)
                test_ids += list(image_ids)

            for i, _id in enumerate(test_ids):
                if _id in ids_to_probs:
                    ids_to_probs[_id].append(np.copy(test_probs[i]))
                else:
                    ids_to_probs[_id] = [np.copy(test_probs[i])]
        
            coord.request_stop()
            coord.join(threads)
            sess.close()

    with open('probs.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
        for img_name, probs in ids_to_probs.items():
            writer.writerow([img_name] + list(np.mean(probs, axis=0)))
        # for img_name, probs in zip(test_ids, test_probs):
        #     writer.writerow([img_name] + list(probs))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train a model')

    parser.add_argument('-data_dir', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/data/test_stg1', dest='data_dir')
    parser.add_argument('-start_from', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/checkpoints/cv/', dest='start_from')

    parser.add_argument('-batch_size', type=int, default=64, dest='batch_size')
    parser.add_argument('-lr', type=float, default=3e-4, dest='lr')

    args = parser.parse_args()
    test_net(args)