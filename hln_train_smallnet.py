# hln_train.py
import numpy as np

from hln import *
from dataset import *

def main():
    batch_size = 8
    ds = Dataset()
    #ds.load(CIFAR10, '../cifar-10-batches-py/')
    ds.load(CIFAR100, '../cifar-100-python/')
    ds.set_batch_size(batch_size)
    net = HybridLearningNet(
        x_dim=[32,32,3], 
        y_dim=100, 
        dims=[
        [64,3,1],
        [3,3,1],
        [64,3,1],
        [3,3,1]
        ], 
        print_every_n_batch = 100, 
        save_every_n_batch = 10000,
        learning_rate = 0.01,
        batch_size=batch_size)
    
    '''t_beg = time.clock()
    for i in xrange(1000):
        x, y = ds.train_batch()
        print 'expected: '+ str(y) + ' predicted: ' + str(net.predict_label(x))
    t_end = time.clock()
    print str(batch_size*1000.0/(t_end-t_beg)) + ' FPS.'
    '''
    net.save('../model/')
    mode = 0
    hist_len = 100
    uerr_hist = np.ones([hist_len])
    
    # hybrid learning method
    for i in xrange(62500000): # at most 1000 epoch
        x, y = ds.train_batch()
        if mode==0:
            loss, serr, uerr, train_acc = net.train(x)
            uerr_hist[i%hist_len] = uerr
            if np.mean(uerr_hist)<1e-4:
                mode = 1
                print '===========begin to train in hybrid manner============'
        else:
            loss, serr, uerr, train_acc = net.train(x, y)
    net.finish()
    
main()

