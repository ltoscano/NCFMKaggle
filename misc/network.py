import tensorflow as tf
import numpy as np 

OP_SEED = 42 

class Network(object):
    def __init__(self):
        self.layers = {}

    def layer(op):
        def add_layer(self, *args, **kwargs):
            layer = op(self, *args, **kwargs)
            self.layers[kwargs["name"]] = layer
            return layer

        return add_layer

    def load(self, path, sess):
        model = np.load(path).item()

        for layer in model.keys():
            with tf.variable_scope(layer, reuse=True) as scope:
                for subkey in model[layer]:
                    try:
                        var = tf.get_variable(subkey)
                        sess.run(var.assign(model[layer][subkey]))
                    except ValueError:
                        print("ignore -- {}/{}".format(layer, subkey))
                

    def make_var(self, name, shape, initializer, trainable):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    @layer
    def conv(self, bottom, k_h, k_w, c_o, s_h, s_w, name, train):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            c_i = shape[-1]

            weights = self.make_var("weights", [k_h, k_w, c_i, c_o], tf.truncated_normal_initializer(0, stddev=.02, seed=OP_SEED), train)
            biases = self.make_var("biases", [c_o], tf.constant_initializer(0.0), train)

            conv = tf.nn.conv2d(bottom, weights, [1, s_h, s_w, 1], padding="SAME")
            bias = tf.nn.bias_add(conv, biases)
            return tf.nn.relu(bias)

    @layer
    def max_pool(self, bottom, k_h, k_w, s_h, s_w, name):
        return tf.nn.max_pool(bottom, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1],
                              padding="VALID", name=name)

    @layer
    def fc(self, bottom, out_dim, name, train):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            in_dim = 1
            for d in shape[1:]:
                 in_dim *= d
            x = tf.reshape(bottom, [-1, in_dim])

            weights = self.make_var("weights", [in_dim, out_dim], tf.truncated_normal_initializer(0, stddev=.02, seed=OP_SEED), train)
            biases = self.make_var("biases", [out_dim], tf.constant_initializer(0.0), train)

            return tf.nn.bias_add(tf.matmul(x, weights), biases)

    @layer
    def relu(self, bottom, name):
        return tf.nn.relu(bottom, name=name)

    @layer
    def dropout(self, bottom, keep_prob, name):
        return tf.nn.dropout(bottom, keep_prob, name=name, seed=OP_SEED)

if __name__ == '__main__':

    im = np.random.randn(12, 4, 4, 3).astype(np.float32)
    im_tf = tf.placeholder(tf.float32, shape=[None, 4, 4, 3])
    network = Network()
    convolve = network.conv(im, 3, 3, 14, 1, 1, name="conv1", train=True)
    fc = network.fc(convolve, 1000, name="fc1", train=True) 

    print(network.layers)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        out = sess.run(fc, feed_dict={im_tf: im})
        print(out.shape)


            
