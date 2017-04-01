import tensorflow as tf

import numpy
audioArr=numpy.load("test_logS.npy")

#test that file loaded correctly
print (audioArr)
print (audioArr.shape)

flatAudio = audioArr.flatten()

audioCol=numpy.array([flatAudio,flatAudio])
x = tf.placeholder(tf.float32, [None, 128*8484])

audioTens = tf.placeholder(tf.float32)
# genres defined by 0, 1, 2
W = tf.Variable(tf.zeros([128*8484, 3]))
genreTens = tf.Variable(tf.zeros([3]))

y = tf.nn.softmax(tf.matmul(x, W) + genreTens)
y_ = tf.placeholder(tf.float32, [None, 3])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()


tf.global_variables_initializer().run()
ye = numpy.array([[0,0,1],[0,0,1]])

sess.run(train_step, feed_dict={x: audioCol, y_: ye})


correctTest = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
print(sess.run(correctTest,{x: audioCol, y_: ye}))

