import tensorflow as tf 
import numpy as np 

from network import Network 

class VGG16(Network):
    def __init__(self, train, layer=None):
        super(VGG16, self).__init__()
        self.train = train
        self.layer = layer 
        self.build()

    def build(self):
        self.input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

        self.relu1_1 = self.conv(self.input, 3, 3, 64, 1, 1, name="conv1_1", train=self.train)
        self.relu1_2 = self.conv(self.relu1_1, 3, 3, 64, 1, 1, name="conv1_2", train=self.train)
        self.pool1 = self.max_pool(self.relu1_2, 2, 2, 2, 2, name='pool1')

        self.relu2_1 = self.conv(self.pool1, 3, 3, 128, 1, 1, name='conv2_1', train=self.train)
        self.relu2_2 = self.conv(self.relu2_1, 3, 3, 128, 1, 1, name='conv2_2', train=self.train)
        self.pool2 = self.max_pool(self.relu2_2, 2, 2, 2, 2, name='pool2')

        self.relu3_1 = self.conv(self.pool2, 3, 3, 256, 1, 1, name='conv3_1', train=self.train)
        self.relu3_2 = self.conv(self.relu3_1, 3, 3, 256, 1, 1, name='conv3_2', train=self.train)
        self.relu3_3 = self.conv(self.relu3_2, 3, 3, 256, 1, 1, name='conv3_3', train=self.train)
        self.pool3 = self.max_pool(self.relu3_3, 2, 2, 2, 2, name='pool3')

        self.relu4_1 = self.conv(self.pool3, 3, 3, 512, 1, 1, name='conv4_1', train=self.train)
        self.relu4_2 = self.conv(self.relu4_1, 3, 3, 512, 1, 1, name='conv4_2', train=self.train)
        self.relu4_3 = self.conv(self.relu4_2, 3, 3, 512, 1, 1, name='conv4_3', train=self.train)
        self.pool4 = self.max_pool(self.relu4_3, 2, 2, 2, 2, name='pool4')

        self.relu5_1 = self.conv(self.pool4, 3, 3, 512, 1, 1, name='conv5_1', train=self.train)
        self.relu5_2 = self.conv(self.relu5_1, 3, 3, 512, 1, 1, name='conv5_2', train=self.train)
        self.relu5_3 = self.conv(self.relu5_2, 3, 3, 512, 1, 1, name='conv5_3', train=self.train)
        if self.layer == 'conv5_3':
            self.output = self.relu5_3
            return 

        self.pool5 = self.max_pool(self.relu5_3, 2, 2, 2, 2, name='pool5')
        self.fc6 = self.fc(self.pool5, 4096, name='fc6', train=self.train)
        if self.layer == 'fc6':
            self.output = self.fc6
            return 

        self.relu6 = self.relu(self.fc6, name='relu6')
        # self.relu6 = tf.nn.dropout(self.relu6, 0.5, name='drop6')
        self.fc7 = self.fc(self.relu6, 4096, name='fc7', train=self.train)
        if self.layer == 'fc7':
            self.output = self.fc7
            return 

        self.relu7 = tf.nn.relu(self.fc7, name='relu7')
        # self.relu7 = tf.nn.dropout(self.relu7, 0.5, name='drop7')
        self.fc8 = self.fc(self.relu7, 1000, name='fc8', train=self.train)
        self.output = tf.nn.softmax(self.fc8, name='prob')

if __name__ == '__main__':

    im = np.random.randn(12, 224, 224, 3).astype(np.float32)
    vgg16 = VGG16(False, layer='fc6')
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        vgg16.load("VGG_imagenet.npy", sess)
        # sess.run(init_op)
        out = sess.run(vgg16.output, feed_dict={vgg16.input: im})
        print(out.shape)