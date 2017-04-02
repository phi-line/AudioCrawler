import tensorflow as tf
from random import randint
import numpy
audioArr=numpy.load("test_logS.npy")
dict = {"dnb":numpy.array([1,0,0]),"house":numpy.array([0,1,0]),"dstep":numpy.array([0,0,1])}
class smRegAI(object):
    def __init__(self,timeLen):
        self.sess= tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, [None, timeLen])
        self.W = tf.Variable(tf.zeros([timeLen, 3]))
        self.genreTens = tf.Variable(tf.zeros([3]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.genreTens)
        self.testAudio = []
        self.testResults =[]
        self.y_ = tf.placeholder(tf.float32, [None, 3])

    def __del__(self):
        self.sess.close()


#test that file loaded correctly
    def teachMe(self,SongArr,Genre):
       # flatAudio = SongArr.flatten()

        if randint(0,2)==1 :
            self.testAudio+=[SongArr]
            self.testResults += [dict[Genre]]

      #  audioCol=numpy.array([flatAudio])


        audioTens = tf.placeholder(tf.float32)
# genres defined by 0, 1, 2
#a[3] = vector = 1D
#[array]  = a[1][3] = 2D


        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



        tf.global_variables_initializer().run()


        self.sess.run(train_step, feed_dict={self.x: SongArr, self.y_:[dict[Genre]]})
    def predict(self,songArr):
       # flatAudio=songArr.flatten()
        prediction=tf.argmax(self.y,1)
        #print (self.sess.run(self.y,feed_dict={self.x: [flatAudio]}))
        print (prediction.eval(feed_dict={self.x: [songArr]}))
    def checkAccuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(self.sess.run(accuracy, feed_dict={self.x: self.testAudio, self.y_: self.testResults}))

class smRegAlog(object):
    def __init__(self,timeLen):
        #0=mel, 1=perc,2=harm, 3= chroma
        self.numpties = []
        for num in range(0,3):
            self.numpties.append(smRegAI(timeLen))
    def teachAI(self,songTuple):
        for num in range(0,3):
            self.numpties[num].teachMe(songTuple[num],songTuple[4])     # teach all teh numpties!!! :D
    def predict(self,songTuple):
        for num in range(0,3):
            self.numpties[num].predict(songTuple[num])     #
#test = smRegAI()
#test.teachMe(audioArr,"dstep")
#test.teachMe(audioArr,"dstep")
#test.predict(audioArr)
#test.checkAccuracy()
#audioArr
