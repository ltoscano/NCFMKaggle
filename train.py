import tensorflow as tf 
import argparse 
import os 
import json
import numpy as np 

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
        self.labels = tf.placeholder(tf.int32, shape=[None])

        self.l1 = self.fc(self.base.output, 1000, name="l1", train=True)
        self.l1reldrop = self.dropout(self.relu(self.l1, name="l1relu"), 0.5, name="l1drop")

        self.l2 = self.fc(self.l1reldrop, self.num_classes, name="l2", train=True)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.l2, self.labels, name="cross_entropy"))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

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

def train_net(args):
    tf.set_random_seed(42)
    np.random.seed(42)

    loader = DataLoader(args.batch_size, args.input_folder, args.input_json, args.val_split)
    vgg16 = VGG16(trainable=args.finetune_cnn, layer='fc6')
    net = fishTest(vgg16, args.lr)

    save_name = os.path.join(args.save_folder, args.save_name)
    with open(save_name + '.json', 'w') as f:
        json.dump(vars(args), f)

    iter_ix = 0
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        vgg16.load(args.cnn_model, sess)

        if len(args.start_from) > 0:
            optimistic_restore(sess, args.start_from)

        while True:
            # training iteration
            img, label, _ = loader.next_batch(0)
            loss, _ = sess.run([net.loss, net.train_step], feed_dict={net.input: img, net.labels: label})

            epoch = float(iter_ix)*args.batch_size/loader.train_size
            print('epoch {:.2f}: {:.2f}'.format(epoch, loss))

            # evaluate validation loss 
            if iter_ix % args.save_every == 0:
                loss_sum = 0
                loss_eval = 0
                while True:
                    val_img, val_label, wrap = loader.next_batch(1)
                    loss = sess.run(net.loss, feed_dict={net.input: val_img, net.labels: val_label})
                    loss_sum += loss
                    loss_eval = loss_eval + 1
                    if wrap:
                        break

                val_loss = loss_sum / loss_eval
                print('validation loss: {:.2f}'.format(val_loss))

                if iter_ix == 0:
                    best_loss = val_loss 

                if iter_ix > 0 and val_loss < best_loss:
                    best_loss = val_loss

                    print('saving checkpoint...')
                    ckpt_name = save_name + '-{:.2f}'.format(best_loss)
                    file_save = saver.save(sess, ckpt_name, global_step=iter_ix, write_meta_graph=False)
                    print(file_save)

            iter_ix = iter_ix + 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train a model')

    parser.add_argument('-input_json', type=str, default='fish.json', dest='input_json')
    parser.add_argument('-input_folder', type=str, default='train_proc/', dest='input_folder')
    parser.add_argument('-cnn_model', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/VGG_imagenet.npy', dest='cnn_model')
    parser.add_argument('-start_from', type=str, default='', dest='start_from')

    parser.add_argument('-batch_size', type=int, default=64, dest='batch_size')
    parser.add_argument('-lr', type=float, default=3e-4, dest='lr')
    parser.add_argument('-val_split', type=float, default=.2, dest='val_split')
    parser.add_argument('-cnn_layer', type=str, default='fc7', dest='cnn_layer', 
                            help='layer from which to start cnn finetuning (empty -- no finetuning)')
    parser.add_argument('-finetune_cnn_layer', type=str, default='', dest='finetune_cnn_layer', 
                        help='layer from which to start cnn finetuning (empty -- no finetuning)')

    parser.add_argument('-save_name', type=str, default='linear-r1', dest='save_name')
    parser.add_argument('-save_folder', type=str, default='/scratch/cluster/vsub/ssayed/NCFMKaggle/checkpoints', dest='save_folder')
    parser.add_argument('-save_every', type=int, default=50, dest='save_every')

    args = parser.parse_args()
    train_net(args)