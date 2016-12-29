import tensorflow as tf 

from misc.DataLoader import DataLoader 
from misc.VGG16 import VGG16 
from misc.network import Network

class fishTest(Network):
    def __init__(self, base_model):
        super(fishTest, self).__init__()
        self.base = base_model
        self.num_classes = 8
        self.input = base_model.input 
        self.build()

    def build(self):
        self.labels = tf.placeholder(tf.int32, shape=[None])

        self.l1 = self.fc(self.base.output, 1000, name="l1", train=True)
        self.l1reldrop = self.dropout(self.relu(self.l1, name="l1relu"), 0.5, name="l1drop")

        self.l2 = self.fc(self.l1reldrop, self.num_classes, name="l2", train=True)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.l2, self.labels, name="cross_entropy"))
        self.train_step = tf.train.AdamOptimizer(4e-4).minimize(self.loss)

loader = DataLoader(16, 'train_proc/', 'fish.json')
vgg16 = VGG16(False, layer='fc6')
net = fishTest(vgg16)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # vgg16.load("misc/VGG_imagenet.npy", sess)
    while True:
        img, label = loader.next_batch()
        loss, _ = sess.run([net.loss, net.train_step], feed_dict={net.input: img, net.labels: label})
        print(loss)