''' HLN: Hybrid Learning Network
    X<---[Conv#r1]<-----------[Conv#r2]<------------- ... ----[Conv#rN]-------
            /(shake)            /(shake)                                     /(shake)
    X--->[Conv#1]------------>[Conv#2]--------------> ... --->[Conv#N]-->classifier
            \                    \                               \ 
             --->[ASR]--->0       --->[ASR]--->0                  --->[ASR]--->0
    Where, Conv is Convolution Op, 
    ASR is Activation Sparsity Regularization
    Using shake-shake regularization, too.
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class HybridLearningNet(object):
    def __init__(
            self, 
            dims, 
            print_every_n_batch, 
            learning_rate
            ):
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
        for i in xrange(len(dims)):
            dims[i]
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
        _, serr, uerr, train_acc, sparsity = self.sess.run(
                [self.ops['opt'], 
                    self.ops['serr'],
                    self.ops['uerr'], 
                    self.ops['train_acc'],
                    self.ops['sparsity']], 
                feed_dict=[self.x:x, self.y:y])
        if self.counter%self.print_every_n_batch==0:
            print('#' + str(self.counter) + 
                    ' serr:' + str(serr) + 
                    ' uerr:' + str(uerr) + 
                    ' train_acc:' + str(train_acc) + 
                    ' sparsity:' + str(sparsity))
            pass
