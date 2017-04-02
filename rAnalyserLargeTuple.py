import tensorflow as tf
from random import randint
import numpy
audioArr=numpy.load("test_logS.npy")
dict = {"dnb":[1,0,0],"house":[0,1,0],"dstep":[0,0,1]}
class smRegAI(object):
    def __init__(self,timeLen):
        self.sess= tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, [None, timeLen])
        self.W = tf.Variable(tf.zeros([timeLen, 3]))
        self.genreTens = tf.Variable(tf.zeros([3]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.genreTens)
        #self.testAudio
        #self.testResults
        self.y_ = tf.placeholder(tf.float32, [None, 3])

    def __del__(self):
        self.sess.close()


#test that file loaded correctly
    def teachMe(self,SongArr,Genre):
       # flatAudio = SongArr.flatten()


      #  audioCol=numpy.array([flatAudio])


        audioTens = tf.placeholder(tf.float32)
# genres defined by 0, 1, 2
#a[3] = vector = 1D
#[array]  = a[1][3] = 2D
        hotGenre=numpy.full([40,3],dict[Genre[0]])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



        tf.global_variables_initializer().run()


        self.sess.run(train_step, feed_dict={self.x: SongArr, self.y_:hotGenre})
    def predict(self,songArr):
       # flatAudio=songArr.flatten()
        prediction=tf.argmax(self.y,1)
        #print (self.sess.run(self.y,feed_dict={self.x: [flatAudio]}))
        print (prediction.eval(feed_dict={self.x: songArr}))
   # def checkAccuracy(self):
    #    correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
     #   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      #  print(self.sess.run(accuracy, feed_dict={self.x: self.testAudio, self.y_: self.testResults}))

class smRegAlog(object):
    def __init__(self,dataTuple):
        #0=mel, 1=perc,2=harm, 3= chroma
        self.numpties = []
        for num in range(0,3):
            self.numpties.append(smRegAI(dataTuple[2][0]))
        for num in range(0,3):
            self.numpties[num].teachMe(dataTuple[0][num],dataTuple[1])     # teach all teh numpties!!! :D
    def predict(self,songTuple):
        for num in range(0,3):
            self.numpties[num].predict(songTuple[num])

test = smRegAI(128*8484)
flatAudi=audioArr.flatten()
test.teachMe([flatAudi],["dstep"])
test.teachMe([flatAudi],["dstep"])
test.predict([flatAudi])
#test.checkAccuracy()
#audioArr
