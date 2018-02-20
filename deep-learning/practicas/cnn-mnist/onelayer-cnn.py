import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#Código proporcionado por: Sihan Tabik

#Creamos las variables

batch_size = 100
learning_rate = 0.5
training_epochs = 10
mnist = input_data.read_data_sets("data", one_hot=True)

#First, we create the necessary tensors

#The set of input images of size 28x28 pixels each one>
X = tf.placeholder(tf.float32, [None, 784], name="input")
#The output of of the network, a tensor of 10 elements>
Y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
XX = tf.reshape(X, [-1, 784])

#Then, we define our flowgraph
Y = tf.nn.softmax(tf.matmul(XX, W) + b, name="output")
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

#Finally, we crearte a session to run our flowgraph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run([train_step],feed_dict={X: batch_x,Y_: batch_y})
        print("Epoch: ", epoch)
    print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
    print("done")
