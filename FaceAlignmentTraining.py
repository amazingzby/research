import tensorflow as tf

import numpy as np

from layers import *
#from layers import AffineTransformLayer
#from layers import TransformParamsLayer
#from layers import LandmarkImageLayer
#from layers import LandmarkTransformLayer
#from layers import TransformParamsLayer_test


class FaceAlignmnetTraining(object):
    def __init__(self,initLandmarks,batch_size, nStages):
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
        #l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #l2_losses = tf.add_n(l2_losses)
        total_loss = (loss/norm)#+l2_losses
        return total_loss

    def addDANStage(self,stageIdx,Input,sn_landmarks,trainable,l2_scale):
        prevStage = 'S' + str(stageIdx - 1)
        curStage = 'S' + str(stageIdx)
        
        IMGSIZE = 112
        #CONNNECTION LAYERS OF PREVIOUS STAGE
        sn_transform       = TransformParamsLayer(sn_landmarks,self.initLandmarks)
        sn_img_output      = AffineTransformLayer(Input,sn_transform)

        sn_landmarks_affine= LandmarkTransformLayer(sn_landmarks,sn_transform)
        sn_img_landmarks   = LandmarkImageLayer(sn_landmarks_affine)
        sn_img_landmarks_f = tf.contrib.layers.flatten(sn_img_landmarks,scope=curStage+'/sn_img_landmarks_flatten')
        sn_img_feature     = tf.layers.dense(sn_img_landmarks_f,56*56,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer())
        sn_img_feature     = tf.reshape(sn_img_feature,[-1, 1, 56, 56])
        sn_img_feature     = tf.image.resize_images(sn_img_feature,(IMGSIZE,IMGSIZE),1)#name = prevStage + '/_img_feature')

        #CURRENT STAGE
        with tf.variable_scope(curStage):
            sn_input   = tf.concat([sn_img_output,sn_img_landmarks,sn_img_feature],axis=3)
            sn_input_bn= tf.layers.batch_normalization(sn_input,trainable=trainable)
            sn_conv1_1 = tf.layers.conv2d(sn_input_bn,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            sn_bn1_1   = tf.layers.batch_normalization(sn_conv1_1,trainable=trainable)
            sn_conv1_2 = tf.layers.conv2d(sn_bn1_1,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            sn_bn1_2   = tf.layers.batch_normalization(sn_conv1_2,trainable=trainable)
            sn_pool1   = tf.layers.max_pooling2d(sn_bn1_2,2,2,padding='same')
    
            #layers: conv2_1,bn2_1,conv2_2,bn2_2,pool2
            sn_conv2_1 = tf.layers.conv2d(sn_pool1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            sn_bn2_1   = tf.layers.batch_normalization(sn_conv2_1,trainable=trainable)
            sn_conv2_2 = tf.layers.conv2d(sn_bn2_1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            sn_bn2_2   = tf.layers.batch_normalization(sn_conv2_2,trainable=trainable)
            sn_pool2   = tf.layers.max_pooling2d(sn_bn2_2,2,2,padding='same')
    
            #layers: conv3_1,bn3_1,conv3_2,bn3_2,pool3
            sn_conv3_1 = tf.layers.conv2d(sn_pool2,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            sn_bn3_1   = tf.layers.batch_normalization(sn_conv3_1,trainable=trainable)
            sn_conv3_2 = tf.layers.conv2d(sn_bn3_1,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            sn_bn3_2   = tf.layers.batch_normalization(sn_conv3_2,trainable=trainable)
            sn_pool3   = tf.layers.max_pooling2d(sn_bn3_2,2,2,padding='same')
    
            #layers: conv4_1,bn4_1,conv4_2,bn4_2,pool4
            sn_conv4_1 = tf.layers.conv2d(sn_pool3,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            sn_bn4_1   = tf.layers.batch_normalization(sn_conv4_1,trainable=trainable)
            sn_conv4_2 = tf.layers.conv2d(sn_bn4_1,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            sn_bn4_2   = tf.layers.batch_normalization(sn_conv4_2,trainable=trainable)
            sn_pool4   = tf.layers.max_pooling2d(sn_bn4_2,2,2,padding='same')
     
            scale      = tf.constant(14/float(112))
            sn_pair    = CreatPair(sn_bn4_2,sn_landmarks_affine,scale,self.batch_size)
            #sn_pair_norm=tf.layers.batch_normalization(sn_pair,trainable=trainable)
            sn_pair_fc = tf.layers.dense(sn_pair,256,trainable=trainable)

            sn_pool4_flat = tf.contrib.layers.flatten(sn_pool4,scope=curStage+'/pool4_flat')

            #sn_concat  = tf.concat([sn_pool4_flat,sn_pair],axis=1)
            sn_fc1     = tf.layers.dense(sn_pool4_flat,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer(),
                        trainable=trainable )
            sn_fc1_bn  = tf.layers.batch_normalization(sn_fc1,trainable=trainable)

            sn_pair_concat  = tf.concat([sn_pair_fc,sn_fc1_bn],axis=1)
            sn_output  = tf.layers.dense(sn_pair_concat,136,trainable=trainable)
            sn_landmark= sn_landmarks_affine+sn_output
            sn_output  = LandmarkTransformLayer(sn_landmark ,sn_landmarks_affine,True)
        return sn_output
    
    def createCNN(self,Input,groundTruth,trainable_params,l2_scale):
        trainable = trainable_params[0]
        #undefined params: l2_scale,trainable
        with tf.variable_scope("S1"): 
            #layers: conv1_1,bn1_1,conv1_2,bn1_2,pool1
            s1_conv1_1 = tf.layers.conv2d(Input,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s1_bn1_1   = tf.layers.batch_normalization(s1_conv1_1,trainable=trainable)
            s1_conv1_2 = tf.layers.conv2d(s1_bn1_1,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s1_bn1_2   = tf.layers.batch_normalization(s1_conv1_2,trainable=trainable)
            s1_pool1   = tf.layers.max_pooling2d(s1_bn1_2,2,2,padding='same')

            #layers: conv2_1,bn2_1,conv2_2,bn2_2,pool2
            s1_conv2_1 = tf.layers.conv2d(s1_pool1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s1_bn2_1   = tf.layers.batch_normalization(s1_conv2_1,trainable=trainable)
            s1_conv2_2 = tf.layers.conv2d(s1_bn2_1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s1_bn2_2   = tf.layers.batch_normalization(s1_conv2_2,trainable=trainable)
            s1_pool2   = tf.layers.max_pooling2d(s1_bn2_2,2,2,padding='same')

            #layers: conv3_1,bn3_1,conv3_2,bn3_2,pool3
            s1_conv3_1 = tf.layers.conv2d(s1_pool2,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s1_bn3_1   = tf.layers.batch_normalization(s1_conv3_1,trainable=trainable)
            s1_conv3_2 = tf.layers.conv2d(s1_bn3_1,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s1_bn3_2   = tf.layers.batch_normalization(s1_conv3_2,trainable=trainable)
            s1_pool3   = tf.layers.max_pooling2d(s1_bn3_2,2,2,padding='same')

            #layers: conv4_1,bn4_1,conv4_2,bn4_2,pool4
            s1_conv4_1 = tf.layers.conv2d(s1_pool3,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s1_bn4_1   = tf.layers.batch_normalization(s1_conv4_1,trainable=trainable)
            s1_conv4_2 = tf.layers.conv2d(s1_bn4_1,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer() )
            s1_bn4_2   = tf.layers.batch_normalization(s1_conv4_2,trainable=trainable)
            s1_pool4   = tf.layers.max_pooling2d(s1_bn4_2,2,2,padding='same')

        
            s1_pool4_flat = tf.contrib.layers.flatten(s1_pool4)
            s1_pool4_drop = tf.layers.dropout(s1_pool4_flat,0.5,training=trainable)

            s1_fc1        = tf.layers.dense(s1_pool4_drop,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer(),
                            trainable=trainable )
            s1_fc1_bn     = tf.layers.batch_normalization(s1_fc1,trainable=trainable)
            s1_output     = tf.layers.dense(s1_fc1_bn,136,trainable=trainable)

        sn_landmarks  = tf.math.add(s1_output,self.initLandmarks,name='S1/add')

        for i in range(1,self.nStages):
            sn_landmarks=self.addDANStage(i+1,Input,sn_landmarks,trainable_params[1],l2_scale)
        sn_cost      = tf.reduce_mean(self.normRmse(groundTruth,sn_landmarks))
        print(sn_landmarks.shape)
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'stage')):
        S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(sn_cost)#,\
        #                   var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'s1'))
        return sn_cost,S2_Optimizer,sn_landmarks
        
