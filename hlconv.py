# hlconv.py
# hybrid learning convolutional neural network on CIFAR-100,
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image as pi
import _tkinter as tk

''' activation polarizing function '''
def tf_polarize_op(x):
  delta_max = tf.reduce_max(x, axis=[1,???])

'''
tf_hlconv_op : create a op in tensorflow graph to implement hybrid learning
  convolutional layer.
x: input tensor
sw: the supervised learning weights
uw: the unsupervised learning weights
'''
def tf_hlconv_op(x, sw, uw):
  sy = tf.nn.conv2d(
    x,
    sw,
    strides=[1,1,1,1],
    padding='SAME',
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None)
  uy = tf.

class HLConvNet(object):
  def __init__(self, dims):
    self.dims = dims
    for i in xrange(len(dims)):
      pass
