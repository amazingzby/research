import tensorflow as tf

import numpy as np

from layers import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer
#from layers import *
#from layers import AffineTransformLayer
#from layers import TransformParamsLayer
#from layers import LandmarkImageLayer
#from layers import LandmarkTransformLayer
#from layers import TransformParamsLayer_test
IMGSIZE = 112
N_LANDMARK = 68
lr_rate = 0.5
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
        l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_losses = tf.add_n(l2_losses)
        total_loss = (loss/norm)+l2_losses
        return total_loss
    """
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
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            sn_bn1_1   = tf.layers.batch_normalization(sn_conv1_1,trainable=trainable)
            sn_conv1_2 = tf.layers.conv2d(sn_bn1_1,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            sn_bn1_2   = tf.layers.batch_normalization(sn_conv1_2,trainable=trainable)
            sn_pool1   = tf.layers.max_pooling2d(sn_bn1_2,2,2,padding='same')
    
            #layers: conv2_1,bn2_1,conv2_2,bn2_2,pool2
            sn_conv2_1 = tf.layers.conv2d(sn_pool1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            sn_bn2_1   = tf.layers.batch_normalization(sn_conv2_1,trainable=trainable)
            sn_conv2_2 = tf.layers.conv2d(sn_bn2_1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            sn_bn2_2   = tf.layers.batch_normalization(sn_conv2_2,trainable=trainable)
            sn_pool2   = tf.layers.max_pooling2d(sn_bn2_2,2,2,padding='same')
    
            #layers: conv3_1,bn3_1,conv3_2,bn3_2,pool3
            sn_conv3_1 = tf.layers.conv2d(sn_pool2,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            sn_bn3_1   = tf.layers.batch_normalization(sn_conv3_1,trainable=trainable)
            sn_conv3_2 = tf.layers.conv2d(sn_bn3_1,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            sn_bn3_2   = tf.layers.batch_normalization(sn_conv3_2,trainable=trainable)
            sn_pool3   = tf.layers.max_pooling2d(sn_bn3_2,2,2,padding='same')
    
            #layers: conv4_1,bn4_1,conv4_2,bn4_2,pool4
            sn_conv4_1 = tf.layers.conv2d(sn_pool3,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            sn_bn4_1   = tf.layers.batch_normalization(sn_conv4_1,trainable=trainable)
            sn_conv4_2 = tf.layers.conv2d(sn_bn4_1,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            sn_bn4_2   = tf.layers.batch_normalization(sn_conv4_2,trainable=trainable)
            sn_pool4   = tf.layers.max_pooling2d(sn_bn4_2,2,2,padding='same')
     
            scale      = tf.constant(14/float(112))
            sn_pair    = CreatPair(sn_bn4_2,sn_landmarks_affine,scale,self.batch_size)
            #sn_pair_norm=tf.layers.batch_normalization(sn_pair,trainable=trainable)
            sn_pair_fc = tf.layers.dense(sn_pair,256,trainable=trainable)

            sn_pool4_flat = tf.contrib.layers.flatten(sn_pool4,scope=curStage+'/pool4_flat')

            #sn_concat  = tf.concat([sn_pool4_flat,sn_pair],axis=1)
            sn_fc1     = tf.layers.dense(sn_pool4_flat,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer(),
                        trainable=trainable,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            sn_fc1_bn  = tf.layers.batch_normalization(sn_fc1,trainable=trainable)

            sn_pair_concat  = tf.concat([sn_pair_fc,sn_fc1_bn],axis=1)
            sn_output  = tf.layers.dense(sn_pair_concat,136,trainable=trainable)
            sn_landmark= sn_landmarks_affine+sn_output
            sn_output  = LandmarkTransformLayer(sn_landmark ,sn_landmarks_affine,True)
        return sn_output
    """    
    def createCNN(self,Input,groundTruth,l2_scale,stage):
        #trainable = trainable_params[0]
        #trainable = tf.placeholder(tf.bool)
        #S2_isTrain = tf.placeholder(tf.bool)
        trainable =True
        S2_isTrain =True
        Ret_dict   = {}
        #Ret_dict['S1_isTrain'] = trainable
        #Ret_dict['S2_isTrain'] = S2_isTrain
        #undefined params: l2_scale,trainable
        with tf.variable_scope("S1"): 
            #layers: conv1_1,bn1_1,conv1_2,bn1_2,pool1
            s1_conv1_1 = tf.layers.conv2d(Input,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_bn1_1   = tf.layers.batch_normalization(s1_conv1_1,trainable=trainable)
            s1_conv1_2 = tf.layers.conv2d(s1_bn1_1,64,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_bn1_2   = tf.layers.batch_normalization(s1_conv1_2,trainable=trainable)
            s1_pool1   = tf.layers.max_pooling2d(s1_bn1_2,2,2,padding='same')

            #layers: conv2_1,bn2_1,conv2_2,bn2_2,pool2
            s1_conv2_1 = tf.layers.conv2d(s1_pool1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_bn2_1   = tf.layers.batch_normalization(s1_conv2_1,trainable=trainable)
            s1_conv2_2 = tf.layers.conv2d(s1_bn2_1,128,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_bn2_2   = tf.layers.batch_normalization(s1_conv2_2,trainable=trainable)
            s1_pool2   = tf.layers.max_pooling2d(s1_bn2_2,2,2,padding='same')

            #layers: conv3_1,bn3_1,conv3_2,bn3_2,pool3
            s1_conv3_1 = tf.layers.conv2d(s1_pool2,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_bn3_1   = tf.layers.batch_normalization(s1_conv3_1,trainable=trainable)
            s1_conv3_2 = tf.layers.conv2d(s1_bn3_1,256,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_bn3_2   = tf.layers.batch_normalization(s1_conv3_2,trainable=trainable)
            s1_pool3   = tf.layers.max_pooling2d(s1_bn3_2,2,2,padding='same')

            #layers: conv4_1,bn4_1,conv4_2,bn4_2,pool4
            s1_conv4_1 = tf.layers.conv2d(s1_pool3,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_bn4_1   = tf.layers.batch_normalization(s1_conv4_1,trainable=trainable)
            s1_conv4_2 = tf.layers.conv2d(s1_bn4_1,512,3,1,padding='same',activation=tf.nn.relu,trainable=trainable,
                kernel_initializer=tf.glorot_uniform_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_bn4_2   = tf.layers.batch_normalization(s1_conv4_2,trainable=trainable)
            s1_pool4   = tf.layers.max_pooling2d(s1_bn4_2,2,2,padding='same')

        
            s1_pool4_flat = tf.contrib.layers.flatten(s1_pool4)
            s1_fc1        = tf.layers.dense(s1_pool4_flat,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer(),
                            trainable=trainable,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_fc1_bn     = tf.layers.batch_normalization(s1_fc1,trainable=trainable)
            s1_output     = tf.layers.dense(s1_fc1_bn,136,trainable=trainable,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
            s1_landmarks  = tf.math.add(s1_output,self.initLandmarks,name='add')
            s1_cost       = tf.reduce_mean(self.normRmse(groundTruth,s1_landmarks))
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'S1')):
                gloabl_steps = tf.Variable(0, trainable=False) 
                lr = tf.train.exponential_decay(lr_rate,gloabl_steps,9525,0.2,staircase=True)
                #s1_optimizer = tf.train.AdamOptimizer(0.001).minimize(s1_cost,
                #          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"S1"))
                s1_optimizer = tf.train.MomentumOptimizer(lr,0.9).minimize(s1_cost,global_step=gloabl_steps,
                           var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"S1"))
        Ret_dict['s1_landmarks'] = s1_landmarks
        Ret_dict['s1_cost']      = s1_cost
        Ret_dict['s1_optimizer'] = s1_optimizer
        #if stage<=1:
        #    return Ret_dict

        with tf.variable_scope('S2'):
            S2_AffineParam = TransformParamsLayer(s1_landmarks, self.initLandmarks)
            S2_InputImage = AffineTransformLayer(Input, S2_AffineParam)
            S2_InputLandmark = LandmarkTransformLayer(s1_landmarks, S2_AffineParam)
            S2_InputHeatmap = LandmarkImageLayer(S2_InputLandmark)
            
            S2_Feature = tf.reshape(tf.layers.dense(s1_fc1_bn,int((IMGSIZE / 2) * (IMGSIZE / 2)),\
                activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),(-1,int(IMGSIZE / 2),int(IMGSIZE / 2),1))
            S2_FeatureUpScale = tf.image.resize_images(S2_Feature,(IMGSIZE,IMGSIZE),1)
            
            S2_ConcatInput = tf.layers.batch_normalization(tf.concat([S2_InputImage,S2_InputHeatmap,S2_FeatureUpScale],3),\
                training=S2_isTrain)
            S2_Conv1a = tf.layers.batch_normalization(tf.layers.conv2d(S2_ConcatInput,64,3,1,\
                padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
            S2_Conv1b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv1a,64,3,1,\
                padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
            S2_Pool1 = tf.layers.max_pooling2d(S2_Conv1b,2,2,padding='same')
            
            S2_Conv2a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool1,128,3,1,\
                padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
            S2_Conv2b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv2a,128,3,1,\
                padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
            S2_Pool2 = tf.layers.max_pooling2d(S2_Conv2b,2,2,padding='same')       
            
            S2_Conv3a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool2,256,3,1,\
                padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
            S2_Conv3b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv3a,256,3,1,\
                padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
            S2_Pool3 = tf.layers.max_pooling2d(S2_Conv3b,2,2,padding='same')
            
            S2_Conv4a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool3,512,3,1,\
                padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
            S2_Conv4b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv4a,512,3,1,\
                padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
            S2_Pool4 = tf.layers.max_pooling2d(S2_Conv4b,2,2,padding='same')
            
            S2_Pool4_Flat = tf.contrib.layers.flatten(S2_Pool4)
            S2_DropOut = tf.layers.dropout(S2_Pool4_Flat,0.5,training=S2_isTrain)
            
            S2_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S2_DropOut,256,\
                activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
            S2_Fc2 = tf.layers.dense(S2_Fc1,N_LANDMARK * 2)
            
            S2_Ret = LandmarkTransformLayer(S2_Fc2 + S2_InputLandmark,S2_AffineParam, Inverse=True)
            S2_Cost = tf.reduce_mean(self.normRmse(groundTruth,S2_Ret))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'S2')):
                global_steps = tf.Variable(0, trainable=False)
                lr = tf.train.exponential_decay(lr_rate,gloabl_steps,9525,0.2,staircase=True)
                #S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S2_Cost,\
                S2_Optimizer = tf.train.AdamOptimizer(lr).minimize(S2_Cost,global_step=global_steps,
        			var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"S2"))
        Ret_dict['S2_Ret'] = S2_Ret
        Ret_dict['S2_Cost'] = S2_Cost
        Ret_dict['S2_Optimizer'] = S2_Optimizer
        return Ret_dict
    
        #for i in range(1,self.nStages):
        #    sn_landmarks=self.addDANStage(i+1,Input,sn_landmarks,trainable_params[1],l2_scale)
        #sn_cost      = tf.reduce_mean(self.normRmse(groundTruth,sn_landmarks))
        #print(sn_landmarks.shape)
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'stage')):
        #S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(sn_cost)#,\
        #                   var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'s1'))
        #return sn_cost,S2_Optimizer,sn_landmarks
        
