import tensorflow as tf
import sys

def tf_to_complex(real, imag):
  complex_v = tf.complex(real, imag)
  return complex_v

def tf_complex_multiply(a, b):
  # (a_real+a_imag*i) * (b_real+b_imag*i)
  ans = tf.math.multiply(a, b)
  return ans

def check_nan(a, name):
  a_real = tf.math.real(a)
  a_imag = tf.math.imag(a)
  a_real = tf.check_numerics(a_real, name+"_real is nan")
  a_imag = tf.check_numerics(a_imag, name+"_imag is nan")
  a = tf.complex(a_real, a_imag)
  return a

def tf_complex_mul_real(v_com, v_real):
  v_com_real = tf.real(v_com) * v_real
  v_com_imag = tf.imag(v_com) * v_real
  v = tf.complex(v_com_real, v_com_imag)
  return v
