import tensorflow as tf

import numpy
audioArr=numpy.load("test_logS.npy")
dict = {"dnb":numpy.array([1,0,0]),"house":numpy.array([0,1,0]),"dstep":numpy.array([0,0,1])}
class smRegAI(object):
    def __init__(self):
        self.sess= tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, [None, 128 * 8484])
        self.W = tf.Variable(tf.zeros([128 * 8484, 3]))
        self.genreTens = tf.Variable(tf.zeros([3]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.genreTens)

    def __del__(self):
        self.sess.close()


#test that file loaded correctly
    def teachMe(self,SongArr,Genre):
        flatAudio = SongArr.flatten()
        audioCol=numpy.array([flatAudio])
        audioTens = tf.placeholder(tf.float32)
# genres defined by 0, 1, 2
#a[3] = vector = 1D
#[array]  = a[1][3] = 2D

        y_ = tf.placeholder(tf.float32, [None, 3])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(self.y), reduction_indices=[1]))

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



        tf.global_variables_initializer().run()


        self.sess.run(train_step, feed_dict={self.x: audioCol, y_:[dict[Genre]]ok})
    def predict(self,songArr):
        flatAudio=songArr.flatten()
        prediction=tf.argmax(self.y,1)
        #print (self.sess.run(self.y,feed_dict={self.x: [flatAudio]}))
        print (prediction.eval(feed_dict={self.x: [flatAudio]}))

class smRegAlog(object):
    def __init__(self):
        #0=mel, 1=perc,2=harm, 3= chroma
        self.numpties = []
        for num in range(0,3):
            self.numpties.append(smRegAI())
    def teachAI(self,songTuple):
        for num in range(0,3):
            self.numpties[num].teachMe(songTuple[num],songTuple[4])     # teach all teh numpties!!! :D
    def predict(self,songTuple):
        for num in range(0,3):
            self.numpties[num].predict(songTuple[num])     #
test = smRegAI()
test.teachMe(audioArr,"dnb")
test.predict(audioArr)
audioArr
