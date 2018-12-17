import os
import tensorflow as tf
import numpy as np
from  data_process import loadTfrecords
from FaceAlignmentTraining import *
#from models import DAN

#os.environ['CUDA_VISIBLE_DEVICES']='2'
#tf.device('/gpu:2')
STAGE = 1
batch_size = 64

tfrecordsPath = 'data/train.tfrecords'
initmarks     = np.load('data/initlandmark.npy')
initmarks     = tf.constant(initmarks,dtype=tf.float32)



imgs_batch,landmarks_batch = loadTfrecords(tfrecordsPath,batch_size)
#initmarks_batch = tf.ones([136])
#landmarks_batchs = tf.ones([batch_size,136])
#initlandmark = tf.placeholder(tf.float32,shape=(1,136),name='initlandmarkss')
x = tf.placeholder(dtype=tf.float32,shape=(batch_size,112,112,1))
y = tf.placeholder(dtype=tf.float32,shape=(batch_size,136))
training = FaceAlignmnetTraining(initmarks,batch_size,2, [1])
loss,train_step=training.createCNN(x,y)
saver = tf.train.Saver()
#print(model.shape)
#tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
    sess.run(tf.global_variables_initializer())
    print("Starting training......")
    for i in range(100000):
        imgs,labels = sess.run([imgs_batch,landmarks_batch])
        sess.run(train_step,feed_dict={x:imgs,y:labels})
        #train_step.run()
        if(i%1000==0):
            temp = loss.eval(feed_dict={x:imgs,y:labels})
            print(temp)
            saver.save(sess,'model/model_iter',global_step=i)

