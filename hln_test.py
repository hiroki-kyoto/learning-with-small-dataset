# hln_train.py
import numpy as np
import matplotlib.pyplot as plt

from hln import *
from dataset import *

def main():
    batch_size = 1
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
        [3,3,1],
        ], 
        print_every_n_batch = 100, 
        save_every_n_batch = 10000,
        learning_rate = 0.01,
        batch_size=batch_size)
    
    #np.random.seed(8603)
    #print net.predict_label(np.random.uniform(0, 1, [8, 32, 32, 3]))
    '''t_beg = time.clock()
    for i in xrange(1000):
        x, y = ds.train_batch()
        print 'expected: '+ str(y) + ' predicted: ' + str(net.predict_label(x))
    t_end = time.clock()
    print str(batch_size*1000.0/(t_end-t_beg)) + ' FPS.'
    '''
    net.restore('../model/')
    
    for i in xrange(1): # 1000 epoch
        x, _ = ds.train_batch()
        r_x = net.reconstruct_image(x)
        print np.mean(r_x[0,:,:,0])
        print np.mean(r_x[0,:,:,1])
        print np.mean(r_x[0,:,:,2])
        plt.imshow(x[0])
        plt.figure()
        plt.imshow(r_x[0])
        plt.show()
    net.finish()
    
main()

