# dataset.py
# Provide functions to load dataset of CIFAR-100,
import pickle
import numpy as np
from ctypes import *
import time
from PIL import Image as pi

class Dataset(object):
  '''
    Dataset: unifying the dataset APIs with loading, 
    and feeding samples into models.
  '''
  def __init__(self):
    self.RAND_SEQ_LENGTH = 1024*1024
    self.initialized = False
    pass
  
  def ready(self):
    return self.initialized
  
  '''
    ds_open_fn : dataset open function
    ds_path : dataset path
    function : get samples and multi-labels
  '''
  def load(self, ds_open_fn, ds_path):
    (self.train_samples, 
    self.train_labels, 
    self.train_coarse_labels,
    self.test_samples,
    self.test_labels,
    self.test_coarse_labels) = ds_open_fn(ds_path)
    # prepare a random sequence for generating training batches
    self.rand_seq = np.random.randint(0, len(self.train_labels), self.RAND_SEQ_LENGTH)
    self.seq_id = 0
    self.batch_size = 0
    self.initialized = True
  
  ''' set the batch size '''
  def set_batch_size(self, size):
    assert(size <= self.RAND_SEQ_LENGTH)
    assert(self.initialized)
    self.batch_size = size
    self.sample_batch = np.zeros([
      size, 
      self.train_samples.shape[1], 
      self.train_samples.shape[2],
      self.train_samples.shape[3]])
    self.label_batch = np.zeros([size])
  
  ''' generate a batch with given size for training '''
  def train_batch(self):
    assert(self.initialized)
    assert(self.batch_size)
    if self.seq_id + self.batch_size > self.RAND_SEQ_LENGTH:
      # regenerate the random sequence
      self.rand_seq = np.random.randint(0, len(self.train_labels), self.RAND_SEQ_LENGTH)
      self.seq_id = 0
    id_beg = self.seq_id
    id_end = self.seq_id + self.batch_size
    self.sample_batch = self.train_samples[self.rand_seq[id_beg : id_end]]
    self.label_batch = self.train_labels[self.rand_seq[id_beg : id_end]]
    self.seq_id += self.batch_size
    return (self.sample_batch, self.label_batch)
  
  ''' generate all sample label pairs for test '''
  def test(self):
    assert(self.initialized)
    return (self.test_samples, self.test_labels)

def CIFAR100(path):
  file_train = path + '/train'
  file_test = path + '/test'
  tools = cdll.LoadLibrary('./libdatasettools.so')
  
  with open(file_train, 'rb') as fo:
    dict = pickle.load(fo)
    n, d = dict[b'data'].shape
    h = 32
    w = 32
    c = 3
    assert(d==h*w*c)
    train_samples = np.zeros([n, h, w, c], dtype=np.float32)
    train_labels = np.zeros([n], dtype=np.float32)
    train_coarse_labels = np.zeros([n], dtype=np.float32)
    data = dict[b'data'];
    
    if not train_samples.flags['C_CONTIGUOUS']:
      train_samples = np.ascontiguous(train_samples, dtype=train_samples.dtype)
    samples_ptr = cast(train_samples.ctypes.data, POINTER(c_float))
    
    if not data.flags['C_CONTIGUOUS']:
      data = np.ascontiguous(data, dtype=data.dtype)
    data_ptr = cast(data.ctypes.data, POINTER(c_uint8))
    
    train_labels = np.array(dict[b'fine_labels'])
    train_coarse_labels = np.array(dict[b'coarse_labels'])
    tools.NCHW2NHWC(samples_ptr, data_ptr, c_int(n), c_int(h), c_int(w), c_int(c))
    
  with open(file_test, 'rb') as fo:
    dict = pickle.load(fo)
    n, d = dict[b'data'].shape
    h = 32
    w = 32
    c = 3
    assert(d==h*w*c)
    test_samples = np.zeros([n, h, w, c], dtype=np.float32)
    test_labels = np.zeros([n], dtype=np.float32)
    test_coarse_labels = np.zeros([n], dtype=np.float32)
    data = dict[b'data'];
    
    if not test_samples.flags['C_CONTIGUOUS']:
      test_samples = np.ascontiguous(test_samples, dtype=test_samples.dtype)
    samples_ptr = cast(test_samples.ctypes.data, POINTER(c_float))
    
    if not data.flags['C_CONTIGUOUS']:
      data = np.ascontiguous(data, dtype=data.dtype)
    data_ptr = cast(data.ctypes.data, POINTER(c_uint8))
    
    test_labels = np.array(dict[b'fine_labels'])
    test_coarse_labels = np.array(dict[b'coarse_labels'])
    tools.NCHW2NHWC(samples_ptr, data_ptr, c_int(n), c_int(h), c_int(w), c_int(c))

  return (train_samples, 
          train_labels, 
          train_coarse_labels,
          test_samples,
          test_labels,
          test_coarse_labels)

def main():
  ds = Dataset()
  ds.load(CIFAR100, '../cifar-100-python')
  ds.set_batch_size(8)
  for i in xrange(10):
    x,y = ds.train_batch()
    #im = pi.fromarray((x[0]*255).astype(np.uint8), mode="RGB")
    #im.show()
  x,y = ds.test()
  im = pi.fromarray((x[2]*255).astype(np.uint8), mode="RGB")
  im.show()
  #print(str(y[2]))

main()
