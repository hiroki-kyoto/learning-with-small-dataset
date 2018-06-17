''' HLN: Hybrid Learning Network
    X<---[Conv#r1]<-----------[Conv#r2]<------------- ... ----[Conv#rN]-------
            /(shake)            /(shake)                         /(shake)
    X--->[Conv#1]------------>[Conv#2]--------------> ... --->[Conv#N]-->classifier
            \                    \                               \ 
             --->[ASR]--->0       --->[ASR]--->0                  --->[ASR]--->0
    Where, conv is convolution op, 
    ASR is Activation Sparsity Regularization
    Using shake-even regularization, too.
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import *

def sparsity_regularizer(x):
    max = tf.reduce_max(x, axis=3)
    sum = tf.reduce_sum(x, axis=3)
    return tf.reduce_mean(sum - max)

def correlation_regularizer(x):
    n = x.shape.as_list()[3]
    x = tf.reshape(x, [-1, n])
    x_std = tf.nn.l2_normalize(x, axis=[1])
    w = tf.random_uniform([n, n])-0.5
    mask = 1 - tf.diag(tf.ones([n]))
    w = w * mask
    w_std = tf.nn.l2_normalize(w, axis=[0])
    r = tf.matmul(x_std, w_std)
    return tf.reduce_mean(tf.reduce_sum(x_std * r, axis=[0]))

def mixed_kernel_regularizer(x):
    return 0.5*tf.nn.l2_loss(x) + 0.5*correlation_regularizer(x)

# Generate new layer with given input layer and settings in tensorflow graph
# returning layer handler
def tf_build_recognitive_layer(inputs, filters, ksizes, rates):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=ksizes,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=rates,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=mixed_kernel_regularizer,
        bias_regularizer=tf.nn.l2_loss,
        activity_regularizer=sparsity_regularizer,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None
    )
    
def tf_build_generative_layer(inputs, filters, ksizes, rates):
    pass
    
class HybridLearningNet(object):
    def __init__(
        self, 
        x_dim, 
        y_dim, 
        dims, 
        print_every_n_batch, 
        learning_rate, 
        batch_size):
        
        assert(len(dims)>0)
        self.counter = 0
        self.print_every_n_batch = print_every_n_batch
        self.learning_rate = learning_rate
        self.ops = dict()
        self.ops['recconv'] = []
        self.ops['genconv'] = []
        self.ops['opt'] = None
        self.ops['serr'] = None
        self.ops['uerr'] = None
        self.ops['tran_acc'] = None
        self.shake_coef = None
        self.batch_size = batch_size
        self.graph = tf.Graph()
        self.x = None
        self.y = None
        self.recon_x = None # reconstructed x
        
        # construct network
        with self.graph.as_default():
            tf.set_random_seed(1587)
            self.sess = tf.Session()
            assert(len(x_dim)==3) # height, width, channels
            self.x = tf.placeholder(
                tf.float32,
                (batch_size, x_dim[0], x_dim[1], x_dim[2]))
            # recognitive layers
            for i in xrange(len(dims)):
                assert(len(dims[i])==3) # num_filters, kernel_size, dilated_rate
                if i==0:
                    self.ops['recconv'].append(
                        tf_build_recognitive_layer(
                            self.x, 
                            dims[i][0], 
                            dims[i][1], 
                            dims[i][2]))
                else:
                    self.ops['recconv'].append(
                        tf_build_recognitive_layer(
                            self.ops['recconv'][-1], 
                            dims[i][0], 
                            dims[i][1], 
                            dims[i][2]))
            assert(type(y_dim)==int)
            self.y = tf.reshape(self.ops['recconv'][-1], [batch_size, -1])
            self.y = tf.layers.dense(
                inputs = self.y,
                units = y_dim,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.nn.l2_loss,
                bias_regularizer=tf.nn.l2_loss,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=None,
                reuse=None)
            # generative layers: 
            # reverse the convolutional part of recognition graph
            self.shake_coef = tf.random_uniform([len(dims)])
            for i in xrange(len(dims)-1):
                i = len(dims) - i - 2
                assert(len(dims[i])==3) # num_filters, kernel_size, dilated_rate
                if i==len(dims)-2:
                    self.ops['genconv'].append(
                        tf_build_generative_layer(
                            self.shake_coef[i+1]*self.ops['recconv'][i+1], 
                            dims[i][0], 
                            dims[i][1], 
                            dims[i][2]))
                else:
                    self.ops['genconv'].append(
                        tf_build_generative_layer(
                            self.shake_coef[i+1]*self.ops['genconv'][-1] + 
                                (1-self.shake_coef[i+1])*self.ops['recconv'][i+1], 
                            dims[i][0], 
                            dims[i][1], 
                            dims[i][2])) 
            # finally reconstruct the input
            self.ops['genconv'].append(
                tf_build_generative_layer(
                    self.shake_coef[0]*self.ops['genconv'][-1] + 
                        (1-self.shake_coef[0])*self.ops['recconv'][0],
                    self.x.shape[1],
                    self.x.shape[2],
                    self.x.shape[3]) #####################
            # saver
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

    def save(self, path):
        pass

    def restore(self, path):
        pass

    def run(self, x):
        assert(x.shape==self.x.shape)
        return np.argmax(self.sess.run(self.y, feed_dict={self.x:x}), axis=1)

    def train(self, x, y):
        assert(x.shape==self.x.shape)
        assert(y.shape==self.y.shape.as_list())
        self.counter += 1
        _, serr, uerr, train_acc = self.sess.run(
                [self.ops['opt'], 
                    self.ops['serr'],
                    self.ops['uerr'], 
                    self.ops['train_acc']], 
                feed_dict={self.x:x, self.y:y})
        if self.counter%self.print_every_n_batch==0:
            print('#' + str(self.counter) + 
                    ' serr:' + str(serr) + 
                    ' uerr:' + str(uerr) + 
                    ' train_acc:' + str(train_acc))

def main():
    ds = Dataset()
    #ds.load(CIFAR10, '../cifar-10-batches-py/')
    ds.load(CIFAR100, '../cifar-100-python/')
    ds.set_batch_size(8)
    
    net = HybridLearningNet(
        x_dim=[32,32,3], 
        y_dim=100, 
        dims=[[8, 3, 3], 
        [16, 3, 3], 
        [8, 3, 3]], 
        print_every_n_batch = 100, 
        learning_rate = 0.01,
        batch_size=8)
    
    #np.random.seed(8603)
    #print net.run(np.random.uniform(0, 1, [8, 32, 32, 3]))
    for i in xrange(1000):
        x, y = ds.train_batch()
        print 'expected: '+ str(y) + ' predicted: ' + str(net.run(x))
    
main()

