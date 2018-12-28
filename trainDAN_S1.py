import os
import time
import tensorflow as tf
import numpy as np
from  data_process import loadTfrecords
from  data_process import loadTfrecords_t
from FaceAlignmentTraining_l2 import *
from scipy.integrate import simps
#from models import DAN

STAGE = 1
batch_size = 64
l2_scale = 1e-6
threshold =0.08
#outputsnap='model/S1_S2_model_iter'
outputsnap='model/S1_model_iter'
auc_step  =50
save_step =1000
total_step = 47625

def LandmarkError(gtLandmarks, resLandmarks, normalization='centers', showResults=False, verbose=False):
    resLandmarks = np.reshape(resLandmarks,(68,2))
    gtLandmarks  = np.reshape(gtLandmarks ,(68,2))
    if normalization == 'centers':
        normDist = np.linalg.norm(np.mean(gtLandmarks[36:42], axis=0) - np.mean(gtLandmarks[42:48], axis=0))
    elif normalization == 'corners':
        normDist = np.linalg.norm(gtLandmarks[36] - gtLandmarks[45])
    elif normalization == 'diagonal':
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


tfrecordsTrain= 'data/train.tfrecords'
tfrecordsTest = 'data/ibus_337.tfrecords'
initmarks     = np.load('data/initlandmark.npy')
initmarks     = tf.constant(initmarks,dtype=tf.float32)

imgs_batch,landmarks_batch = loadTfrecords(tfrecordsTrain,batch_size)
#imgs_test, landmarks_test  = loadTfrecords(tfrecordsTest ,batch_size)
x = tf.placeholder(dtype=tf.float32,shape=(None,112,112,1))
y = tf.placeholder(dtype=tf.float32,shape=(None,136))
training = FaceAlignmnetTraining(initmarks,batch_size,STAGE)
dan=training.createCNN(x,y,l2_scale,STAGE)
saver = tf.train.Saver()
#print(model.shape)
#tf.reset_default_graph()
with tf.Session() as sess:
    curTime=time.clock()
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)
    sess.run(tf.global_variables_initializer())
    print("Starting training......")
    for i in range(total_step):

        imgs,labels = sess.run([imgs_batch,landmarks_batch])
        sess.run(dan['s1_optimizer'],feed_dict={x:imgs,y:labels})#,dan['S1_isTrain']:True,dan['S2_isTrain']:False})
        #sess.run(dan['S2_Optimizer'],feed_dict={x:imgs,y:labels})#,dan['S1_isTrain']:False,dan['S2_isTrain']:True})
        #train_step.run()
        if(i%auc_step==0 or i == (total_step-1)):
            runTime = time.clock()-curTime
            curTime=time.clock()
            loss_s1 = dan['s1_cost'].eval(feed_dict={x:imgs,y:labels})#,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
            #loss_s2 = dan['S2_Cost'].eval(feed_dict={x:imgs,y:labels})#,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
            #print("img value")
            #print(imgs[0:3,0,53:58,0])
            print("step:%s   loss_s1:%.2f  Running time:%.2fs"%(str(i).zfill(6),loss_s1,runTime))
        if(i%auc_step==0 or i == (total_step-1)):
            imgs_test, landmarks_test  = loadTfrecords_t(tfrecordsTest ,1)
            errors_s1 = []
            errors_s2 = []
            for _ in range(337):
                imgsTest,landmarksTest     = sess.run([imgs_test, landmarks_test])
                pred_s1 = sess.run(dan['s1_landmarks'],feed_dict = 
                                        {x:imgsTest,y:landmarksTest})#,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                #pred_s2 = sess.run(dan['S2_Ret'],feed_dict =
                #                        {x:imgsTest,y:landmarksTest})#,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                error_s1 = LandmarkError(pred_s1[0],landmarksTest[0])
                #error_s2 = LandmarkError(pred_s2[0],landmarksTest[0])
                errors_s1.append(error_s1)
                #errors_s2.append(error_s2)
            AUCError(errors_s1, threshold, step=0.0001, showCurve=False)
            #AUCError(errors_s2, threshold, step=0.0001, showCurve=False)
            print('')
			
        if(i%save_step==0 or i == (total_step-1)):
            saver.save(sess,outputsnap,global_step=i)

