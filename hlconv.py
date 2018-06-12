# hlconv.py
# hybrid learning convolutional neural network on CIFAR-100,
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image as pi
import _tkinter as tk

'''
Hybrid Learning Convolution Theory:
  In SOM(self organizing map),
  each vector [m] is a pattern attractor.
  For a given pattern [x], there's only one 
  vector winner whoes response is the 
  maximum among all.
  x is a 4-D input tensor, it will be converted
  into patches with given kernel size.
  In our design, kernels are all in size of 3x3.
  The first will be the normal convolution part,
  The second will be our polarized SOM part.
  The third will be the module merging part.
'''

''' activation polarizing function '''
def tf_polarize_op(delta):
  delta_max = tf.reduce_max(delta, axis=[1,])

'''
tf_hlconv_op : create a op in tensorflow graph to implement hybrid learning
  convolutional layer.
passed-in:
  x: 4-D input tensor, in NHWC format!!!
  knum : kernel number
  ksizes : kernel size
  strides : strides for convolution
  rates : convolutional rates
  padding : padding type, SAME or VALID
returned :
  y : the hlconv result handler
  sw : the supervised learning weights
  uw : the unsupervised learning weights
  sb : the supervised learning bias
'''
def tf_hlconv_op(
  x, 
  knum,
  ksizes, 
  strides, 
  rates, 
  padding
):
  # convert maps into patches
  p = tf.extract_image_patches(
    x, 
    ksizes=ksizes,
    strides=strides,
    rates=rates,
    padding=padding)
  # create parameters for supervised learning
  sw = tf.Variable(
    [knum, ksizes[1]*ksizes[2]*x.shape.as_list[3]], 
    dtype=tf.float32)
  sb = tf.Variable([knum], dtype=tf.float32)
  # create unsupervised learning parameters
  uw = tf.Variable(
    [knum, ksizes[1]*ksizes[2]*x.shape.as_list[3]],
    dtype=tf.float32)
  # apply normal convolution operator
  ps = p.shape.as_list()
  p = tf.reshape(p, [ps[0]*ps[1]*ps[2], ps[3]])
  sy = tf.matmul(p, sw, transpose_b=True) + sb
  # apply SOM operator
  uy = tf_polarize_op(tf.norm(p - uw))
  y = sy * uy
  return y, sw, sb, uw, e

class HLConvNet(object):
  def __init__(self, dims):
    self.dims = dims
    for i in xrange(len(dims)):
      pass
