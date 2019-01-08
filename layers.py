import tensorflow as tf
import numpy as np
import itertools
def  AffineTransformLayer(imgs,r,t):
    r = tf.matrix_inverse(r)
    r = tf.matrix_transpose(r)

    rm = tf.reshape(tf.pad(r, [[0, 0], [0, 0], [0, 1]], mode='CONSTANT'), [-1, 6])
    rm = tf.pad(rm, [[0, 0], [0, 2]], mode='CONSTANT')

    tm = tf.contrib.image.translations_to_projective_transforms(tf.reshape(t, [-1, 2]))
    rtm = tf.contrib.image.compose_transforms(rm, tm)

    return tf.contrib.image.transform(imgs, rtm, "BILINEAR")
"""
def TransformParamsLayer(Input,transform_params,mean,batch_size):
    transformed = Input
    meanShape   = tf.reshape(mean, [68,2])
    def bestFit(transformedShape, meanShape):
        transformedShape = tf.reshape(transformedShape, [68,2])

        dstMean = tf.reduce_mean(meanShape, axis=0)
        srcMean = tf.reduce_mean(transformedShape, axis=0)

        srcCenter = transformedShape - srcMean
        dstCenter = meanShape - dstMean

        srcVec = tf.contrib.layers.flatten(srcCenter)
        dstVec = tf.contrib.layers.flatten(dstCenter)
        norm = (tf.norm(srcVec) ** 2)

        a = tf.tensordot(srcVec, dstVec, 1) / norm

        srcX = tf.reshape(srcVec, [-1, 2])[:, 0]
        srcY = tf.reshape(srcVec, [-1, 2])[:, 1]
        destX = tf.reshape(dstVec, [-1, 2])[:, 0]
        destY = tf.reshape(dstVec, [-1, 2])[:, 1]

        b = tf.reduce_sum(tf.multiply(srcX, destY) - tf.multiply(srcY, destX))
        b = b / norm

        A = tf.reshape(tf.stack([a, b, -b, a]), [2, 2])
        srcMean = tf.tensordot(srcMean, A, 1)
        return tf.concat((tf.reshape(A, [-1,]), dstMean - srcMean), 0)
    return tf.map_fn(lambda s: bestFit(s, transformed), mean,name=transform_params)
"""
def LandmarkTransformLayer(shapes,r,t,Inverse=False):
    if Inverse:
        r = tf.matrix_inverse(r)
        t = tf.matmul(-t,r)
    shapes = tf.matmul(shapes,r) + t
    return shapes
def LandmarkImageLayer(shapes):
    IMGSIZE =112
    Offsets = tf.constant([(x, y) for y in range(IMGSIZE) for x in range(IMGSIZE)],
                        dtype=tf.float32,shape=[1,IMGSIZE,IMGSIZE,2])
    shapes = shapes[:,:,tf.newaxis,tf.newaxis,:]
    value  = Offsets - shapes
    value  = tf.norm(value,axis=-1)
    value  = 1.0 / (tf.reduce_min(value,axis=1) + 1.0)
    value  = tf.expand_dims(value,axis=-1)
    return value
    
"""
def LandmarkInitLayer(s1_output,initLandmarks,s1_landmarks,batch_size):
    #output         = tf.get_variable(name=s1_output,shape=[batch_size,136]) 
    output = s1_output
    #init_landmarks = tf.contrib.layers.flatten(initLandmarks)
    #landmarks_add  = tf.math.add(output,initLandmarks,name=s1_landmarks)
    print(output)
    print(initLandmarks)
    landmarks_add = output + initLandmarks
    return landmarks_add
"""
def TransformParamsLayer(transformed_, meanShape):
    from_shape = tf.reshape(transformed_,[-1,68,2])
    to_shape   = tf.reshape(meanShape,[68,2])

    from_mean = tf.reduce_mean(from_shape,axis=1,keepdims=True)
    to_mean = tf.reduce_mean(to_shape, axis=0, keepdims=True)

    from_centralized = from_shape - from_mean
    to_centralized = to_shape - to_mean

    dot_result = tf.reduce_sum(tf.multiply(from_centralized, to_centralized),
                               axis=[1, 2])
    norm_pow_2 = tf.pow(tf.norm(from_centralized, axis=[1, 2]), 2)
    a = dot_result / norm_pow_2
    b = tf.reduce_sum(tf.multiply(from_centralized[:, :, 0],to_centralized[:, 1]) - 
                      tf.multiply(from_centralized[:, :, 1],to_centralized[:, 0]), 1)/norm_pow_2
    r = tf.reshape(tf.stack([a, b, -b, a], axis=1), [-1, 2, 2])
    t = to_mean - tf.matmul(from_mean, r)
    return r,t

def CreatPair(features,landmarks,scale,batch_size):
    landmarks = landmarks*scale
    landmarks = tf.reshape(landmarks,[batch_size,68,2])
    landmarks = tf.to_int32(landmarks)
    landmarks = tf.clip_by_value(landmarks,0,13)
    index     = tf.constant(np.reshape(np.arange(batch_size),(batch_size,1)),dtype=tf.int32)
    #landmarks = tf.concat([index,landmarks],axis=1)
    #point_features = tf.tf.gather_nd(features,lanmarks)
    point_init_1 = tf.concat([index,landmarks[:,0]],axis=1)
    point_init_2 = tf.concat([index,landmarks[:,1]],axis=1)
    feature_init_1 = tf.gather_nd(features,point_init_1)
    feature_init_2 = tf.gather_nd(features,point_init_2)
    pair_feature = tf.concat([feature_init_1,feature_init_2],axis=1)
    for i in range(68):
        for j in range(i+1,68):
            if i==0 and j==1:
                continue
            point1 = tf.concat([index,landmarks[:,i]],axis=1)
            point2 = tf.concat([index,landmarks[:,j]],axis=1)
            feature1 = tf.gather_nd(features,point1)
            feature2 = tf.gather_nd(features,point2)
            pair_feature += tf.concat([feature1,feature2],axis=1)
    pair_feature = pair_feature/tf.constant(2144,dtype=tf.float32)
    return pair_feature
