import tensorflow as tf 
import argparse 
import os 
import json
import numpy as np 
import csv 

from misc.DataLoader import DataLoader 
from misc.VGG16 import VGG16 
from misc.network import Network

class fishTest(Network):
    def __init__(self, base_model, lr):
        super(fishTest, self).__init__()
        self.base = base_model
        self.lr = lr
        self.num_classes = 8
        self.input = base_model.input 
        self.build()

    def build(self):
        self.l1 = self.relu(self.fc(self.base.output, 1000, name="l1", train=False), name="l1relu")
        self.l2 = self.fc(self.l1, self.num_classes, name="l2", train=False)
        self.probs = tf.nn.softmax(self.l2, name="softmax")

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            try:
                curr_var = tf.get_variable(saved_var_name)
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
            except ValueError:
                continue
    print([var.name for var in restore_vars])
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def test_net(args):
    tf.set_random_seed(42)
    np.random.seed(42)

    loader = DataLoader(args.batch_size, args.input_folder, info_path=None, val_split=0, test=True)
    vgg16 = VGG16(trainable=False, layer='fc6')
    net = fishTest(vgg16, lr=None)

    saver = tf.train.Saver(max_to_keep=1)

    test_names = []
    test_probs = []
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        optimistic_restore(sess, args.model)

        iter_ix = 0
        while True:
            print('{}/{}'.format(iter_ix, loader.train_size))
            val_img, _, wrap, img_names = loader.next_batch()
            probs = sess.run(net.probs, feed_dict={net.input: val_img})
            iter_ix += args.batch_size
            if wrap:
                ix = -1*(iter_ix - loader.train_size)
                test_names += img_names[:ix]
                test_probs += probs.tolist()[:ix]
                break

            test_names += img_names 
            test_probs += probs.tolist()

        print(len(test_names))
        print(len(test_probs))

    with open(args.output_file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for img_name, probs in zip(test_names, test_probs):
            writer.writerow([img_name] + probs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train a model')

    parser.add_argument('-input_folder', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/test_proc/', dest='input_folder')
    parser.add_argument('-model', type=str, default='', dest='model')
    parser.add_argument('-batch_size', type=int, default=64, dest='batch_size')
    parser.add_argument('-output_file', type=str, default='prob.csv', dest='output_file')

    args = parser.parse_args()
    test_net(args)