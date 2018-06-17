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

def sparsity_regularizer(x):
    max = tf.reduce_max(x, axis=3)
    sum = tf.reduce_sum(x, axis=3)
    return tf.reduce_mean(sum - max)

def correlation_regularizer(x):
    x_std = tf.nn.l2_normalize(x, axis=[0,1,2])
    rand_w = tf.random_uniform(
        [x.shape.as_list()[3]], 
        minval=0, 
        maxval=1, 
        dtype=tf.float32
        )-0.5
    w_std = tf.nn.l2_normalize(w, axis=[0])
    return 1 - tf.nn.l2_loss(x - (x_std * rand_w))
    

def mix_kernel_regularizer(x):
    return 0.5*tf.nn.l2_loss(x) + 0.5*correlation_regularizer(x)

# Generate new layer with given input layer and settings in tensorflow graph
# returning layer handlers and and its regularizations
def tf_build_recognition_layer(inputs, filters, ksizes, rates):
    tf.layers.conv2d(
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
    def __init__(self, dims, print_every_n_batch, learning_rate):
        assert(len(dims)>0)
        self.counter = 0
        self.print_every_n_batch = print_every_n_batch
        self.learning_rate = learning_rate
        self.ops = dict()
        self.ops['conv'] = []
        self.ops['full'] = []
        self.ops['opt'] = None
        self.ops['serr'] = None
        self.ops['uerr'] = None
        self.ops['tran_acc'] = None
        self.ops['sparsity'] = None
        self.shake_coef = None
        # construct network
        self.x = tf.placeholder()
        for i in xrange(len(dims)):
            tf_build_recognition_layer()
        pass

    def save(self, path):
        pass

    def restore(self, path):
        pass

    def run(self, x):
        assert(x.shape==self.x.shape.as_list())
        return self.sess.run(self.y, feed_dict=[self.x:x])

    def train(self, x, y):
        assert(x.shape==self.x.shape.as_list())
        assert(y.shape==self.y.shape.as_list())
        self.counter += 1
        _, serr, uerr, train_acc = self.sess.run(
                [self.ops['opt'], 
                    self.ops['serr'],
                    self.ops['uerr'], 
                    self.ops['train_acc']], 
                feed_dict=[self.x:x, self.y:y])
        if self.counter%self.print_every_n_batch==0:
            print('#' + str(self.counter) + 
                    ' serr:' + str(serr) + 
                    ' uerr:' + str(uerr) + 
                    ' train_acc:' + str(train_acc))
            pass
