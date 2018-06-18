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
import time

from dataset import *

######## REMOVE MODULE IN AND OUT ##############
USE_SPARSITY_REGULARIZER = 1
USE_NORM_REGULARIZER = 1
USE_CORRELATION_REGULARIZER = 1
USE_SHAKE_EVEN = 1
USE_SOFT_LOSS = 1
USE_HYBRID_LEARNING = 1
################################################

def sparsity_regularizer(x):
    if USE_SPARSITY_REGULARIZER:
        max = tf.reduce_max(x, axis=3)
        sum = tf.reduce_sum(x, axis=3)
        return tf.reduce_mean(sum - max)
    else:
        return 0

def norm_regularizer(x):
    if USE_NORM_REGULARIZER:
        return tf.reduce_mean(tf.square(x))
    else:
        return 0

def L1_norm_regularizer(x):
    if USE_NORM_REGULARIZER:    
        return tf.reduce_mean(tf.abs(x))
    else:
        return 0

def correlation_regularizer(x):
    if USE_CORRELATION_REGULARIZER:
        n = x.shape.as_list()[3]
        x = tf.reshape(x, [-1, n])
        x_std = tf.nn.l2_normalize(x, axis=[1])
        w = tf.random_uniform([n, n])-0.5
        mask = 1 - tf.diag(tf.ones([n]))
        w = w * mask
        w_std = tf.nn.l2_normalize(w, axis=[0])
        r = tf.matmul(x_std, w_std)
        return tf.reduce_mean(tf.reduce_sum(x_std * r, axis=[0]))
    else:
        return 0

def mixed_kernel_regularizer(x):
    return tf.add(0.5*norm_regularizer(x), 0.5*correlation_regularizer(x))

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
        bias_regularizer=norm_regularizer,
        activity_regularizer=sparsity_regularizer,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        #name='recconv'+i,
        reuse=None
    )
    
def tf_build_generative_layer(inputs, filters, ksizes, rates):
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
        bias_regularizer=norm_regularizer,
        activity_regularizer=sparsity_regularizer,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        #name='genconv'+i,
        reuse=None
    )

def min_max_normalize(x):
    _shape = x.shape.as_list()
    assert(len(_shape)==2)
    x_min = tf.reduce_min(x, axis=[1])
    x_max = tf.reduce_max(x, axis=[1])
    x_scale = tf.maximum(x_max-x_min, 1.0)
    return (x - tf.reshape(x_min,[_shape[0],1]))/tf.reshape(x_scale,[_shape[0],1])

def normalized_soft_loss(x):
    
    pass
    
def reconstruct_loss(x):
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
        self.ops['loss'] = None
        self.shake_coef = None
        self.batch_size = batch_size
        self.graph = tf.Graph()
        self.x = None
        self.y = None
        self.labels = None
        self.groundtruth_labels = None
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
                kernel_regularizer=L1_norm_regularizer,
                bias_regularizer=L1_norm_regularizer,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=None,
                reuse=None)
            self.labels = tf.argmax(self.y, axis=1)
            self.groundtruth_labels = tf.placeholder(tf.int64, [batch_size])
            # generative layers: 
            # reverse the convolutional part of recognition graph
            if USE_SHAKE_EVEN:
                self.shake_coef = tf.random_uniform([len(dims)])
            else:
                self.shake_coef = np.ones([len(dims)])
            for i in xrange(len(dims)-1):
                i = len(dims) - i - 2
                assert(len(dims[i])==3) # num_filters, kernel_size, dilated_rate
                if i==len(dims)-2:
                    self.ops['genconv'].append(
                        tf_build_generative_layer(
                            self.shake_coef[i+1]*self.ops['recconv'][i+1], 
                            dims[i][0], 
                            dims[i+1][1], 
                            dims[i+1][2]))
                else:
                    self.ops['genconv'].append(
                        tf_build_generative_layer(
                            self.shake_coef[i+1]*self.ops['genconv'][-1] + 
                                (1-self.shake_coef[i+1])*self.ops['recconv'][i+1], 
                            dims[i][0], 
                            dims[i+1][1], 
                            dims[i+1][2])) 
            # finally reconstruct the input
            self.recon_x = tf_build_generative_layer(
                    self.shake_coef[0]*self.ops['genconv'][-1] + 
                        (1-self.shake_coef[0])*self.ops['recconv'][0],
                    self.x.shape[3],
                    dims[0][1],
                    dims[0][2])
            assert(self.recon_x.shape.as_list()==self.x.shape)
            # collect regularizers
            regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if USE_SHAKE_EVEN:
                reg_w = tf.random_uniform([len(regs)])*0.5
            else:
                reg_w = np.ones([len(regs)])
            for i in xrange(len(regs)):
                if i==0:
                    self.ops['loss'] = regs[i]*reg_w[i]
                else:
                    self.ops['loss'] += regs[i]*reg_w[i]
            # obtain supervised loss
            if USE_SOFT_LOSS:
                self.ops['serr'] = None
                pass
            else:
                # use cross entropy loss
                self.ops['serr'] = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.groundtruth_labels,
                    logits=self.y,
                    name='serr')
                pass
            # obtain unsupervised loss
            if USE_HYBRID_LEARNING:
                # add unsupervised loss to total loss
                pass
            
            # build optimizer for model
            self.ops['opt'] = None
            # training accuracy
            scores = tf.equal(self.labels, self.groundtruth_labels)
            self.ops['train_acc'] = tf.reduce_mean(tf.cast(scores, tf.float32))
            # saver
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

    def save(self, path):
        
        pass

    def restore(self, path):
        pass

    def run(self, x):
        assert(x.shape==self.x.shape)
        return self.sess.run(self.labels, feed_dict={self.x:x})

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
    batch_size = 8
    ds = Dataset()
    #ds.load(CIFAR10, '../cifar-10-batches-py/')
    ds.load(CIFAR100, '../cifar-100-python/')
    ds.set_batch_size(batch_size)
    net = HybridLearningNet(
        x_dim=[32,32,3], 
        y_dim=100, 
        dims=[[8, 3, 3], 
        [16, 3, 3], 
        [8, 3, 3]], 
        print_every_n_batch = 100, 
        learning_rate = 0.01,
        batch_size=batch_size)
    
    #np.random.seed(8603)
    #print net.run(np.random.uniform(0, 1, [8, 32, 32, 3]))
    t_beg = time.clock()
    for i in xrange(1000):
        x, y = ds.train_batch()
        print 'expected: '+ str(y) + ' predicted: ' + str(net.run(x))
    t_end = time.clock()
    print str(batch_size*1000.0/(t_end-t_beg)) + ' FPS.'
main()

