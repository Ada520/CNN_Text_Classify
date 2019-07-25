#这仅仅是一个测试文件而已，
# 在复现代码的过程中，测试一些小的计算或者调用方法

import tensorflow as tf
import numpy as np
# input=tf.Variable(tf.random_uniform([1,5,5,1]))
# filter=tf.Variable(tf.random_uniform([3,3,1,7]))
# op1=tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding="VALID")
# input=tf.Variable(tf.random_uniform([1,5,5,1]))
# filter=tf.Variable(tf.random_uniform([3,3,1,7]))
# op2=tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding="SAME")
# input=tf.Variable(tf.random_uniform([1,5,5,1]))
# filter=tf.Variable(tf.random_uniform([3,3,1,7]))
# op3=tf.nn.conv2d(input,filter,strides=[1,2,2,1],padding="VALID")
# input=tf.Variable(tf.random_uniform([1,5,5,1]))
# filter=tf.Variable(tf.random_uniform([3,3,1,7]))
# op4=tf.nn.conv2d(input,filter,strides=[1,2,2,1],padding="SAME")
# input=tf.Variable(tf.random_uniform([1,7,7,1]))
# filter=tf.Variable(tf.random_uniform([3,3,1,7]))
# op5=tf.nn.conv2d(input,filter,strides=[1,3,3,1],padding="SAME")
# input=tf.Variable(tf.random_uniform([1,7,7,1]))
# filter=tf.Variable(tf.random_uniform([3,3,1,7]))
# op6=tf.nn.conv2d(input,filter,strides=[1,3,3,1],padding="VALID")

a=tf.constant([
        [[1.0,2.0,3.0,4.0],
        [5.0,6.0,7.0,8.0],
        [8.0,7.0,6.0,5.0],
        [4.0,3.0,2.0,1.0]],
        [[4.0,3.0,2.0,1.0],
         [8.0,7.0,6.0,5.0],
         [1.0,2.0,3.0,4.0],
         [5.0,6.0,7.0,8.0]]
    ])
a1=tf.Variable(tf.random_uniform([1,2,2,1]))
a2=tf.Variable(tf.random_uniform([1,2,2,1]))
b=tf.concat([a1,a2],3)
a3=tf.Variable(tf.random_uniform([2,3,3,1]))
a4=tf.Variable(tf.random_uniform([1,3,3,2]))
a5=tf.Variable(tf.random_uniform([2,3,3,2]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(op1.shape)
    # print(op2.shape)
    # print(op3.shape)
    # print(op4.shape)
    # print(op5.shape)
    # print(op6.shape)
    sess.run(b)
    print(b)

















































