
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = tf.Variable(tf.random_uniform([784,10],-1,1))
b = tf.Variable(tf.zeros([10]))

NUM_CORES=2

x = tf.placeholder(tf.float32, (None,784))

y = tf.nn.softmax(tf.matmul(x,w)+b)

with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,intra_op_parallelism_threads=NUM_CORES)) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y,{x: np.random.random((100,784))}))
