import tensorflow as tf

import numpy as np

from layers import *
#from layers import AffineTransformLayer
#from layers import TransformParamsLayer
#from layers import LandmarkImageLayer
#from layers import LandmarkInitLayer
#from layers import LandmarkTransformLayer
#from layers import TransformParamsLayer_test


class FaceAlignmnetTraining(object):
    def __init__(self,initLandmarks,batch_size, nStages,stagesToTrain,confidenceLayer=False):
        self.batch_size = batch_size
        self.nStages = nStages
        self.initLandmarks = initLandmarks
        self.landmarkPatchSize = 16

    def normRmse(self,GroundTruth,Prediction):
        N_LANDMARK = 68
        Gt = tf.reshape(GroundTruth, [-1, N_LANDMARK, 2])
        Pt = tf.reshape(Prediction, [-1, N_LANDMARK, 2])
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)
        norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)
        return loss/norm

    def addDANStage(self,stageIdx,Input,sn_landmarks):
        prevStage = 's' + str(stageIdx - 1)
        curStage = 's' + str(stageIdx)
        
        IMGSIZE = 112
        #CONNNECTION LAYERS OF PREVIOUS STAGE
        sn_transform       = TransformParamsLayer_test(prevStage + '_landmarks',prevStage + '_transform_params',self.initLandmarks,self.batch_size)
        sn_img_output      = AffineTransformLayer(Input,prevStage + '_transform_params',prevStage + '_img_output',self.batch_size)

        sn_landmarks_affine= LandmarkTransformLayer(sn_landmarks,sn_transform,prevStage + '_landmarks_affine',self.batch_size)
        sn_img_landmarks   = LandmarkImageLayer(prevStage + '_landmarks_affine',self.landmarkPatchSize,self.batch_size)
        sn_img_landmarks_f = tf.contrib.layers.flatten(sn_img_landmarks,scope=curStage+'/sn_img_landmarks_flatten')
        sn_img_feature     = tf.layers.dense(sn_img_landmarks_f,56*56,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer())
        sn_img_feature     = tf.reshape(sn_img_feature,[-1, 1, 56, 56])
        sn_img_feature     = tf.image.resize_images(sn_img_feature,(IMGSIZE,IMGSIZE),1)#name = prevStage + '/_img_feature')

        #CURRENT STAGE
        sn_input   = tf.layers.batch_normalization(tf.concat([sn_img_output,sn_img_landmarks,sn_img_feature],axis=3),name=curStage +'/input') 
        sn_conv1_1 = tf.layers.batch_normalization(tf.layers.conv2d(sn_input  ,64,3,1,padding='same',\
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name=curStage+'/conv1_1')
        sn_conv1_2 = tf.layers.batch_normalization(tf.layers.conv2d(sn_conv1_1,64,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name=curStage+'/conv1_2')
        sn_pool1   = tf.layers.max_pooling2d(sn_conv1_2,2,2,padding='same',name=curStage+'/pool1')

        sn_conv2_1 = tf.layers.batch_normalization(tf.layers.conv2d(sn_pool1  ,128,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name=curStage+'/conv2_1')
        sn_conv2_2 = tf.layers.batch_normalization(tf.layers.conv2d(sn_conv2_1,128,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name=curStage+'/conv2_2')
        sn_pool2   = tf.layers.max_pooling2d(sn_conv2_2,2,2,padding='same',name=curStage+'/pool2')

        sn_conv3_1 = tf.layers.batch_normalization(tf.layers.conv2d(sn_pool2  ,256,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name=curStage+'/conv3_1')
        sn_conv3_2 = tf.layers.batch_normalization(tf.layers.conv2d(sn_conv3_1,256,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name=curStage+'/conv3_2')
        sn_pool3   = tf.layers.max_pooling2d(sn_conv3_2,2,2,padding='same',name=curStage+'/pool3')

        sn_conv4_1 = tf.layers.batch_normalization(tf.layers.conv2d(sn_pool3  ,512,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name=curStage+'/conv4_1')
        sn_conv4_2 = tf.layers.batch_normalization(tf.layers.conv2d(sn_conv4_1,512,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name=curStage+'/conv4_2')
        sn_pool4   = tf.layers.max_pooling2d(sn_conv4_2,2,2,padding='same',name=curStage+'/pool4')
        
        scale      = tf.constant(14/float(112))
        sn_pair    = CreatPair(sn_conv4_2,sn_landmarks_affine,scale,self.batch_size)

        sn_pool4_flat = tf.contrib.layers.flatten(sn_pool4,scope=curStage+'/pool4_flat')

        sn_concat  = tf.concat([sn_pool4_flat,sn_pair],axis=1)
        #sn_fc1_dropout= tf.layers.dropout(sn_pool4_flat,0.5,name=curStage+'/fc1_dropout')
        sn_fc1        = tf.layers.batch_normalization(tf.layers.dense(sn_concat,256,activation=tf.nn.relu,
            kernel_initializer=tf.glorot_uniform_initializer()),name = curStage+'/fc1')
        sn_output     = tf.layers.dense(sn_fc1,136,name = curStage+'/output')
        sn_landmark   = sn_landmarks_affine + sn_output
        sn_landmark   = LandmarkTransformLayer(sn_landmark ,sn_landmarks_affine,curStage + '_landmarks',True)
        return sn_landmark
    
    def createCNN(self,Input,groundTruth):
        #Input = tf.placeholder(tf.float32,shape=(-1,height,width),name='input')
        s1_conv1_1 = tf.layers.batch_normalization(tf.layers.conv2d(Input,64,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name='s1/conv1_1')
        s1_conv1_2 = tf.layers.batch_normalization(tf.layers.conv2d(s1_conv1_1,64,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name='s1/conv1_2')
        s1_pool1   = tf.layers.max_pooling2d(s1_conv1_2,2,2,padding='same',name='s1/pool1')

        s1_conv2_1 = tf.layers.batch_normalization(tf.layers.conv2d(s1_pool1  ,128,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name='s1/conv2_1')
        s1_conv2_2 = tf.layers.batch_normalization(tf.layers.conv2d(s1_conv2_1,128,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name='s1/conv2_2')
        s1_pool2   = tf.layers.max_pooling2d(s1_conv2_2,2,2,padding='same',name='s1/pool2')

        s1_conv3_1 = tf.layers.batch_normalization(tf.layers.conv2d(s1_pool2  ,256,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name='s1/conv3_1')
        s1_conv3_2 = tf.layers.batch_normalization(tf.layers.conv2d(s1_conv3_1,256,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name='s1/conv3_2')
        s1_pool3   = tf.layers.max_pooling2d(s1_conv3_2,2,2,padding='same',name='s1/pool3')

        s1_conv4_1 = tf.layers.batch_normalization(tf.layers.conv2d(s1_pool3  ,512,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name='s1/conv4_1')
        s1_conv4_2 = tf.layers.batch_normalization(tf.layers.conv2d(s1_conv4_1,512,3,1,padding='same',
            activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),name='s1/conv4_2')
        s1_pool4   = tf.layers.max_pooling2d(s1_conv4_2,2,2,padding='same',name='s1/pool4')
        
        s1_pool4_flat = tf.contrib.layers.flatten(s1_pool4,scope='s1/pool4_flat')
        s1_fc1_dropout= tf.layers.dropout(s1_pool4_flat,0.5,name='s1/fc1_dropout')
        s1_fc1        = tf.layers.batch_normalization(tf.layers.dense(s1_fc1_dropout,256,activation=tf.nn.relu,
            kernel_initializer=tf.glorot_uniform_initializer()),name = 's1/fc1')
        s1_output     = tf.layers.dense(s1_fc1,136,name = 's1/output')
        #sn_landmarks = LandmarkInitLayer('s1/output',self.initLandmarks,'s1/landmarks',self.batch_size)
        #sn_landmarks = LandmarkInitLayer(s1_output,self.initLandmarks,'s1/landmarks',self.batch_size)
        sn_landmarks  = s1_output+self.initLandmarks
        for i in range(1,self.nStages):
            sn_landmarks=self.addDANStage(i+1,Input,sn_landmarks)
        sn_cost      = tf.reduce_mean(self.normRmse(groundTruth,sn_landmarks))
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'stage')):
        S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(sn_cost)#,\
        #                   var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'s1'))
        return sn_cost,S2_Optimizer
        
