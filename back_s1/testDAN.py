import numpy as np
import tensorflow as tf
from  data_process import loadTfrecords
from FaceAlignmentTraining import *
from scipy.integrate import simps
STAGE = 1
batch_size = 1
total_img  = 554
testSet    = "data/commonSet.npz"
outputsnap='model/S1_model_iter-47624'
tfrecordsPath = 'data/train.tfrecords'
initmarks     = np.load('data/initlandmark.npy')
initmarks     = tf.constant(initmarks,dtype=tf.float32)
stdDevImg     = np.load("data/stdDevImg.npy")
meanImg       = np.load("data/meanImg.npy")

l2_scale = 5e-5
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

imgs_batch,landmarks_batch = loadTfrecords(tfrecordsPath,batch_size)
x = tf.placeholder(dtype=tf.float32,shape=(None,112,112,1))
y = tf.placeholder(dtype=tf.float32,shape=(None,136))

training = FaceAlignmnetTraining(initmarks,batch_size,STAGE)
dan=training.createCNN(x,y,l2_scale,STAGE)
saver = tf.train.Saver()

with tf.Session() as sess:
    dataset       = np.load(testSet)
    dataImgs      = (dataset["imgs"]-meanImg)/stdDevImg
    #print(dataImgs.shape)
    dataLandmarks = dataset["gtLandmarks"]
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,outputsnap)
    errors = []
    for i in range(total_img):
        img  = dataImgs[i].reshape((1,112,112,1))
        label= dataLandmarks[i].reshape((1,136))
        if STAGE <= 1:
            pred = sess.run(dan['s1_landmarks'],feed_dict = {x:img,y:label})
        if STAGE == 2:
            pred = sess.run(dan['S2_Ret'],feed_dict = {x:img,y:label})
        error = LandmarkError(pred[0],label[0])
        errors.append(error)
    AUCError(errors,0.08,step=0.0001, showCurve=False)
