import os
import time
import tensorflow as tf
import numpy as np
from  data_process import loadTfrecords
from  data_process import loadTfrecords_t
from FaceAlignmentTraining import FaceAlignmnetTraining
from scipy.integrate import simps
#from models import DAN

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

STAGE = 2
batch_size = 64
threshold =0.08
#outputsnap='model/S1_S2_model_iter'
outputsnap='model/S2_model_iter'
auc_step  =100
save_step =1000
total_step = 42863

def LandmarkError(gtLandmarks, resLandmarks, normalization='centers', showResults=False, verbose=False):
    resLandmarks = np.reshape(resLandmarks,(68,2))
    gtLandmarks  = np.reshape(gtLandmarks ,(68,2))
    #if normalization == 'centers':
    #    normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
    #elif normalization == 'corners':
    #    normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
    #elif normalization == 'diagonal':
    height, width = np.max(gtLandmarks, axis=0) - np.min(gtLandmarks, axis=0)
    normDist = np.sqrt(width ** 2 + height ** 2)
    error = np.mean(np.sqrt(np.sum((gtLandmarks - resLandmarks)**2, axis=1))) / normDist
    return error
def AUCError(errors, failureThreshold, step=0.0001, showCurve=False):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced =  [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]
    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]
    print('AUC @ %.3f  : %.3f'%(failureThreshold, AUC))
    print('Failure rate: %.3f'%(failureRate))
    print('Meanerror   : %.3f'%(np.mean(errors)))

tfrecordsTrain= 'data/train.tfrecords'
tfrecordsTest = 'data/validation.tfrecords'
initmarks     = np.load('data/initlandmark.npy')
initmarks     = tf.constant(initmarks,dtype=tf.float32)

imgs_batch,landmarks_batch = loadTfrecords(tfrecordsTrain,batch_size)
#imgs_test, landmarks_test  = loadTfrecords(tfrecordsTest ,batch_size)
x = tf.placeholder(dtype=tf.float32,shape=(None,112,112,1))
y = tf.placeholder(dtype=tf.float32,shape=(None,136))
training = FaceAlignmnetTraining(initmarks,batch_size,STAGE)
dan=training.createCNN(x,y,STAGE)
saver = tf.train.Saver()
#print(model.shape)
#tf.reset_default_graph()
with tf.Session() as sess:
    curTime=time.clock()
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='S1'))
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'./model/S1_model_iter-0')
    print("Starting training......")
    for i in range(total_step):

        imgs,labels = sess.run([imgs_batch,landmarks_batch])
        sess.run(dan['S2_Optimizer'],feed_dict={x:imgs,y:labels})
        if(i%auc_step==0 or i == (total_step-1)):
            runTime = time.clock()-curTime
            curTime=time.clock()
            loss_s2 = dan['S2_Cost'].eval(feed_dict={x:imgs,y:labels})
            #r = dan['r'].eval(feed_dict={x:imgs,y:labels})
            print("step:%s   loss_s2:%.4f  Running time:%.2fs"%(str(i).zfill(6),loss_s2,runTime))
        if(i%auc_step==0 or i == (total_step-1)):
            imgs_test, landmarks_test  = loadTfrecords_t(tfrecordsTest ,1)
            errors_s2 = []
            for _ in range(100):
                imgsTest,landmarksTest     = sess.run([imgs_test, landmarks_test])
                pred_s2 = sess.run(dan['S2_Ret'],feed_dict =
                                        {x:imgsTest,y:landmarksTest})
                error_s2 = LandmarkError(pred_s2[0],landmarksTest[0])
                errors_s2.append(error_s2)
            AUCError(errors_s2, threshold, step=0.0001, showCurve=False)
            print(sess.run(dan['S2_Cost'],feed_dict ={x:imgsTest,y:landmarksTest}))
            print('')
			
        if(i%save_step==0 or i == (total_step-1)):
            saver.save(sess,outputsnap,global_step=i)

