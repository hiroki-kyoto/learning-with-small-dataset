# dataset.py
# Provide functions to load dataset of CIFAR-100,
import pickle
import numpy as np
from ctypes import *
import time

class Dataset(object):
  '''
    Dataset: unifying the dataset APIs with loading, 
    and feeding samples into models.
  '''
  def __init__(self):
    pass
  
  '''
    ds_open_fn : dataset open function
    ds_path : dataset path
    function : get samples and multi-labels
  '''
  def load(self, ds_open_fn, ds_path):
    self.samples, self.labels, self.coarse_labels = ds_open_fn(ds_path)

def CIFAR100(path):
  file_train = path + '/train'
  file_test = path + '/test'
  
  with open(file_train, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    tools = cdll.LoadLibrary('./libdatasettools.so')
    
    n, d = dict[b'data'].shape
    h = 32
    w = 32
    c = 3
    assert(d==h*w*c)
    samples = np.zeros([n, h, w, c])
    labels = np.zeros([n])
    coarse_labels = np.zeros([n])
    data = dict[b'data'];
    
    if not samples.flags['C_CONTIGUOUS']:
      samples = np.ascontiguous(samples, dtype=samples.dtype)
    samples_ptr = cast(samples.ctypes.data, POINTER(c_float))
    
    if not data.flags['C_CONTIGUOUS']:
      data = np.ascontiguous(data, dtype=data.dtype)
    data_ptr = cast(data.ctypes.data, POINTER(c_uint8))
    
    labels = np.array(dict[b'fine_labels'])
    coarse_labels = np.array(dict[b'coarse_labels'])
    
    t_begin = time.start()
    tools.NCHW2NHWC(samples_ptr, data_ptr, c_int(n), c_int(h), c_int(w), c_int(c))
    t_end = time.stop()
    print('clib costs: ' + str(t_end-t_begin) + ' seconds.')
    
    t_begin = time.start()
    for i in range(n):
      for j in range(h):
        for k in range(w):
          for l in range(c):
            samples[i,j,k,l] = dict[b'data'][i, l*1024+j*32+k]
    t_end = time.stop()
    print('python costs: ' + str(t_end-t_begin) + ' seconds.')
    
    return samples, labels, coarse_labels

def main():
  ds = Dataset()
  ds.load(CIFAR100, './cifar-100-python')

main()