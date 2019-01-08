import tensorflow as tf

import numpy as np

#from layers import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer
from layers import *
#from layers import AffineTransformLayer
#from layers import TransformParamsLayer
#from layers import LandmarkImageLayer
#from layers import LandmarkTransformLayer
#from layers import TransformParamsLayer_test
IMGSIZE = 112
N_LANDMARK = 68
lr_rate = 0.1

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

class FaceAlignmnetTraining(object):
    def __init__(self,initLandmarks,batch_size, nStages):
        self.batch_size = batch_size
        self.nStages = nStages
        self.initLandmarks = initLandmarks
        self.landmarkPatchSize = 16
    """
    def normRmse(self,GroundTruth,Prediction,scope):
        N_LANDMARK = 68
        Gt = tf.reshape(GroundTruth, [-1, N_LANDMARK, 2])
        Pt = tf.reshape(Prediction, [-1, N_LANDMARK, 2])
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)
        #norm = tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.reduce_max(groundtruth,1),tf.reduce_min(groundtruth,1)),-1))
        norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)
        l2_losses = tf.losses.get_regularization_losses(scope=scope)
        #l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_losses = tf.add_n(l2_losses)
        total_loss = (loss/norm)#+l2_losses
        return total_loss
    """
    def normRmse_S2(self,GroundTruth,Prediction):
        N_LANDMARK = 68
        Gt = tf.reshape(GroundTruth, [-1, N_LANDMARK, 2])
        Pt = tf.reshape(Prediction, [-1, N_LANDMARK, 2])
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), -1)), -1)
        norm = tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.reduce_max(Pt,1),tf.reduce_min(Pt,1)),-1))
        #norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)
        total_loss = (loss/norm)
        return total_loss
    def createCNN(self,Input,groundTruth,stage):
        #trainable = trainable_params[0]
        #trainable = tf.placeholder(tf.bool)
        #S2_isTrain = tf.placeholder(tf.bool)
        trainable_s1 = False
        trainable_s2 = False
        if stage == 1:
            trainable_s1 = True
        if stage == 2:
            trainable_s2 = True
        print("s1 dropout:")
        print(trainable_s1)
        print("s2 dropout:")
        print(trainable_s2)
        Ret_dict   = {}

        with tf.variable_scope("S1"): 
            #layers: conv1_1,bn1_1,conv1_2,bn1_2,pool1
            s1_conv1_1 = tf.layers.conv2d(Input,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s1,
                kernel_initializer=tf.glorot_uniform_initializer())
            s1_bn1_1   = tf.layers.batch_normalization(s1_conv1_1,trainable=trainable_s1,axis=-1,scale=True,
						 momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s1_conv1_2 = tf.layers.conv2d(s1_bn1_1,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s1,
                kernel_initializer=tf.glorot_uniform_initializer())
            s1_bn1_2   = tf.layers.batch_normalization(s1_conv1_2,trainable=trainable_s1,axis=-1,scale=True,
						 momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s1_pool1   = tf.layers.max_pooling2d(s1_bn1_2,2,2,padding='same')

            #layers: conv2_1,bn2_1,conv2_2,bn2_2,pool2
            s1_conv2_1 = tf.layers.conv2d(s1_pool1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s1,
                kernel_initializer=tf.glorot_uniform_initializer())
            s1_bn2_1   = tf.layers.batch_normalization(s1_conv2_1,trainable=trainable_s1,axis=-1,scale=True,
						 momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s1_conv2_2 = tf.layers.conv2d(s1_bn2_1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s1,
                kernel_initializer=tf.glorot_uniform_initializer())
            s1_bn2_2   = tf.layers.batch_normalization(s1_conv2_2,trainable=trainable_s1,axis=-1,scale=True,
						 momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s1_pool2   = tf.layers.max_pooling2d(s1_bn2_2,2,2,padding='same')

            #layers: conv3_1,bn3_1,conv3_2,bn3_2,pool3
            s1_conv3_1 = tf.layers.conv2d(s1_pool2,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s1,
                kernel_initializer=tf.glorot_uniform_initializer())
            s1_bn3_1   = tf.layers.batch_normalization(s1_conv3_1,trainable=trainable_s1,axis=-1,scale=True,
						 momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s1_conv3_2 = tf.layers.conv2d(s1_bn3_1,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s1,
                kernel_initializer=tf.glorot_uniform_initializer())
            s1_bn3_2   = tf.layers.batch_normalization(s1_conv3_2,trainable=trainable_s1,axis=-1,scale=True,
						 momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s1_pool3   = tf.layers.max_pooling2d(s1_bn3_2,2,2,padding='same')

            #layers: conv4_1,bn4_1,conv4_2,bn4_2,pool4
            s1_conv4_1 = tf.layers.conv2d(s1_pool3,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s1,
                kernel_initializer=tf.glorot_uniform_initializer())
            s1_bn4_1   = tf.layers.batch_normalization(s1_conv4_1,trainable=trainable_s1,axis=-1,scale=True,
						 momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s1_conv4_2 = tf.layers.conv2d(s1_bn4_1,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s1,
                kernel_initializer=tf.glorot_uniform_initializer())
            s1_bn4_2   = tf.layers.batch_normalization(s1_conv4_2,trainable=trainable_s1,axis=-1,scale=True,
						 momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s1_pool4   = tf.layers.max_pooling2d(s1_bn4_2,2,2,padding='same')

        
            s1_pool4_flat = tf.contrib.layers.flatten(s1_pool4)
            s1_dropout    = tf.layers.dropout(s1_pool4_flat,0.5,training=trainable_s1)
            s1_fc1        = tf.layers.dense(s1_dropout,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer(),
                            trainable=trainable_s1)
            s1_fc1        = tf.layers.batch_normalization(s1_fc1,trainable=trainable_s1,axis=-1,scale=True,
							momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s1_output     = tf.layers.dense(s1_fc1,136,activation=None)
            s1_landmarks  = s1_output+self.initLandmarks
            s1_cost       = tf.reduce_mean(self.normRmse_S2(groundTruth,s1_landmarks))
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'S1')):
                s1_optimizer = tf.train.AdamOptimizer(0.001).minimize(s1_cost,
                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"S1"))
        Ret_dict['s1_landmarks'] = s1_landmarks
        Ret_dict['s1_cost']      = s1_cost
        Ret_dict['s1_optimizer'] = s1_optimizer

        with tf.variable_scope("S2"):
            s1_landmarks  = tf.reshape(s1_landmarks,[-1,68,2])
            r,t           = TransformParamsLayer(s1_landmarks, self.initLandmarks)
            S2_img_output = AffineTransformLayer(Input,r,t)
            S2_landmarks_affine = LandmarkTransformLayer(s1_landmarks,r,t)
            S2_img_landmarks    = LandmarkImageLayer(S2_landmarks_affine)

            S2_img_feature      = tf.layers.dense(s1_fc1,56*56,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer())
            S2_img_feature      = tf.reshape(S2_img_feature,[-1,56,56,1])
            S2_img_feature      = tf.image.resize_images(S2_img_feature,[IMGSIZE,IMGSIZE])
            
            S2_inputs = tf.concat([S2_img_output,S2_img_landmarks,S2_img_feature],axis=3)
            S2_inputs = tf.layers.batch_normalization(S2_inputs,trainable=trainable_s2,axis=-1,scale=True,
                        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s2_conv1_1 = tf.layers.conv2d(S2_inputs,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s2,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s2_bn1_1   = tf.layers.batch_normalization(s2_conv1_1,trainable=trainable_s2,axis=-1,scale=True,
                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s2_conv1_2 = tf.layers.conv2d(s2_bn1_1,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s2,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s2_bn1_2   = tf.layers.batch_normalization(s2_conv1_2,trainable=trainable_s2,axis=-1,scale=True,
                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s2_pool1   = tf.layers.max_pooling2d(s2_bn1_2,2,2)

            #layers: conv2_1,bn2_1,conv2_2,bn2_2,pool2
            s2_conv2_1 = tf.layers.conv2d(s2_pool1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s2,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s2_bn2_1   = tf.layers.batch_normalization(s2_conv2_1,trainable=trainable_s2,axis=-1,scale=True,
                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s2_conv2_2 = tf.layers.conv2d(s2_bn2_1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s2,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s2_bn2_2   = tf.layers.batch_normalization(s2_conv2_2,trainable=trainable_s2,axis=-1,scale=True,
                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s2_pool2   = tf.layers.max_pooling2d(s2_bn2_2,2,2)

            #layers: conv3_1,bn3_1,conv3_2,bn3_2,pool3
            s2_conv3_1 = tf.layers.conv2d(s2_pool2,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s2,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s2_bn3_1   = tf.layers.batch_normalization(s2_conv3_1,trainable=trainable_s2,axis=-1,scale=True,
                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s2_conv3_2 = tf.layers.conv2d(s2_bn3_1,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s2,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s2_bn3_2   = tf.layers.batch_normalization(s2_conv3_2,trainable=trainable_s2,axis=-1,scale=True,
                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s2_pool3   = tf.layers.max_pooling2d(s2_bn3_2,2,2)

            #layers: conv4_1,bn4_1,conv4_2,bn4_2,pool4
            s2_conv4_1 = tf.layers.conv2d(s2_pool3,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s2,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s2_bn4_1   = tf.layers.batch_normalization(s2_conv4_1,trainable=trainable_s2,axis=-1,scale=True,
                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s2_conv4_2 = tf.layers.conv2d(s2_bn4_1,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable_s2,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s2_bn4_2   = tf.layers.batch_normalization(s2_conv4_2,trainable=trainable_s2,axis=-1,scale=True,
                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
            s2_pool4   = tf.layers.max_pooling2d(s2_bn4_2,2,2)

            s2_pool4_flat = tf.contrib.layers.flatten(s2_pool4)
            s2_dropout    = tf.layers.dropout(s2_pool4_flat,0.5,training=trainable_s2)
            s2_fc1        = tf.layers.dense(s2_pool4_flat,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer())
            s2_fc1_bn     = tf.layers.batch_normalization(s2_fc1,trainable=trainable_s2,axis=-1,scale=True,
                            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,fused=True)
 
            S2_Fc2 = tf.layers.dense(s2_fc1_bn,N_LANDMARK * 2,activation=None)
            
            #S2_landmarks_affine = tf.reshape(S2_landmarks_affine,[-1,136])
            S2_Fc2 = tf.reshape(S2_Fc2,[-1,68,2])+S2_landmarks_affine
            S2_Ret = LandmarkTransformLayer(S2_Fc2,r,t,Inverse=True)
            S2_Ret = tf.reshape(S2_Ret,[-1,136])
            S2_Cost = tf.reduce_mean(self.normRmse_S2(groundTruth,S2_Ret))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'S2')):
                S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S2_Cost,
                               var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'S2'))
        Ret_dict['S2_Ret'] = S2_Ret
        Ret_dict['S2_Cost'] = S2_Cost
        Ret_dict['S2_Optimizer'] = S2_Optimizer
        return Ret_dict 
