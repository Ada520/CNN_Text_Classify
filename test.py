#这仅仅是一个测试文件而已，
# 在复现代码的过程中，测试一些小的计算或者调用方法

import tensorflow as tf
import numpy as np
input=tf.Variable(tf.random_uniform([10,5,5,5]))
filter=tf.Variable(tf.random_uniform([3,3,5,7]))
op=tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding="VALID")
#op:(1,3,3,1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(op)














































