import tensorflow as tf
import numpy as np
import itertools
def  AffineTransformLayer(Input,transform_params,img_output,batch_size):
    IMGSIZE     = 112
    inImg       = Input
    transParams = tf.get_variable(transform_params,[batch_size,6])
    Pixels = tf.constant(np.array([(x, y) for x in range(IMGSIZE) for y in range(IMGSIZE)],
                             dtype=np.float32), shape=[IMGSIZE, IMGSIZE, 2])
    A = tf.reshape(transParams[:, 0:4], [-1, 2, 2])
    T = tf.reshape(transParams[:, 4:6], [-1, 1, 2])

    A = tf.matrix_inverse(A)
    T = tf.matmul(-T, A)

    T = tf.reverse(T, (-1,))
    A = tf.matrix_transpose(A)

    def affine_transform(I,A,T):
        I = tf.reshape(I, [IMGSIZE, IMGSIZE])

        SrcPixels = tf.matmul(tf.reshape(Pixels, [IMGSIZE * IMGSIZE, 2]), A) + T
        SrcPixels = tf.clip_by_value(SrcPixels, 0, IMGSIZE - 2)
        
        outPixelsMinMin = tf.to_float(tf.to_int32(SrcPixels))
        dxdy = SrcPixels - outPixelsMinMin
        dx = dxdy[:, 0]
        dy = dxdy[:, 1]

        outPixelsMinMin = tf.reshape(tf.to_int32(outPixelsMinMin), [IMGSIZE * IMGSIZE, 2])
        outPixelsMaxMin = tf.reshape(outPixelsMinMin + [1, 0], [IMGSIZE * IMGSIZE, 2])
        outPixelsMinMax = tf.reshape(outPixelsMinMin + [0, 1], [IMGSIZE * IMGSIZE, 2])
        outPixelsMaxMax = tf.reshape(outPixelsMinMin + [1, 1], [IMGSIZE * IMGSIZE, 2])

        OutImage = (1 - dx) * (1 - dy) * tf.gather_nd(I, outPixelsMinMin) + dx * (1 - dy) * tf.gather_nd(I, outPixelsMaxMin) \
                           + (1 - dx) * dy * tf.gather_nd(I, outPixelsMinMax) + dx * dy * tf.gather_nd(I, outPixelsMaxMax)
        return tf.reshape(OutImage, [IMGSIZE, IMGSIZE, 1])
    return tf.map_fn(lambda args: affine_transform(args[0], args[1], args[2]), (inImg, A, T),dtype=tf.float32, name=img_output)

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

def LandmarkTransformLayer(landmarks, transform_params,landmarks_affine,batch_size,Inverse=False):
    landmark_t      = landmarks
    transformParams = transform_params
    N_Landmark = 68
    A = tf.reshape(transformParams[:, 0:4], [-1, 2, 2])
    T = tf.reshape(transformParams[:, 4:6], [-1, 1, 2])
    landmark = tf.reshape(landmark_t, [-1, N_Landmark, 2])

    if Inverse:
        A = tf.matrix_inverse(A)
        T = tf.matmul(-T, A)
    output = tf.reshape(tf.matmul(landmark, A) + T, [-1, N_Landmark * 2],name='landmarks_affine')
    return output
HalfSize=8
IMGSIZE =112
Offsets = tf.constant(np.array(list(itertools.product(range(-HalfSize, HalfSize), \
    range(-HalfSize, HalfSize))), dtype=np.int32), shape=(16, 16, 2))
def LandmarkImageLayer(landmarks_affine,img_landmarks,batch_size):
    Landmarks = tf.get_variable(landmarks_affine,shape=[batch_size,136])
    def draw_landmarks(L):
        def draw_landmarks_helper(Point):
            intLandmark = tf.to_int32(Point)
            locations = Offsets + intLandmark
            dxdy = Point - tf.to_float(intLandmark)
            offsetsSubPix = tf.to_float(Offsets) - dxdy
            vals = 1 / (1 + tf.norm(offsetsSubPix, axis=2))
            img = tf.scatter_nd(locations, vals, shape=(IMGSIZE, IMGSIZE))
            return img
        #reverse作用：x轴和y轴交换
        Landmark = tf.reverse(tf.reshape(L, [-1,2]), [-1])
        # Landmark = tf.reshape(L, (-1, 2))
        Landmark = tf.clip_by_value(Landmark, HalfSize, IMGSIZE - 1 - HalfSize)
        # Ret = 1 / (tf.norm(tf.map_fn(DoIn,Landmarks),axis = 3) + 1)
        Ret = tf.map_fn(draw_landmarks_helper, Landmark)
        Ret = tf.reshape(tf.reduce_max(Ret, axis=0), [IMGSIZE, IMGSIZE, 1])
        return Ret
    return tf.map_fn(draw_landmarks, Landmarks)

def LandmarkInitLayer(s1_output,initLandmarks,s1_landmarks,batch_size):
    #output         = tf.get_variable(name=s1_output,shape=[batch_size,136]) 
    output = s1_output
    #init_landmarks = tf.contrib.layers.flatten(initLandmarks)
    #landmarks_add  = tf.math.add(output,initLandmarks,name=s1_landmarks)
    print(output)
    print(initLandmarks)
    landmarks_add = output + initLandmarks
    return landmarks_add

def TransformParamsLayer_test(transformed_,output, meanShape,batch_size):
    # transformed(source) shape:  64 * 68 * 2
    # meanshape(destination) shape: 68 * 2
    transformed = tf.get_variable(transformed_,shape=[batch_size,136])
    transformed = tf.reshape(transformed,[batch_size,68,2])
    meanShape   = tf.reshape(meanShape,[68,2])
    destination = meanShape
    source = transformed
    
    destMean = tf.reduce_mean(destination, axis=0)  # shape: 1 * 2
    srcMean = tf.reduce_mean(source, axis=1)        # shape: 64 * 2
    srcMean = tf.reshape(srcMean,[batch_size,1,2])
    srcVec = source - srcMean                       # shape: 64 * 68 * 2
    destVec = destination - destMean                # shape: 68 * 2
    
    srcVec_ = tf.reshape(srcVec,[batch_size, 136])
    destVec_ = tf.reshape(destVec,[136])
    
    src_dot_dest = srcVec_ * destVec_       # shape: 64 * 1
    src_dot_dest = tf.reduce_sum(src_dot_dest,axis=1)
    
    srcNorm = tf.norm(srcVec_, ord=2, axis=1) ** 2   # shape: 64 * 1
    a = src_dot_dest / srcNorm
    
    src_1 = srcVec[:, :, 0]                          # shape: 64 * 68
    src_2 = srcVec[:, :, 1]
    
    dest_1 = destVec[:, 0]                           # shape: 68 
    dest_2 = destVec[:, 1]

    add_mut = src_1* dest_2                          # shape: 64 * 1
    sub_mut = src_2* dest_1
        
    b = add_mut - sub_mut                            # shape: 64 * 1
    b = tf.reduce_sum(b,axis=1)
    b = b / srcNorm
    
    gap_mean = destMean - srcMean                    # shape: 64 * 2
    gap_mean = tf.reshape(gap_mean,[batch_size,2])
    a=tf.reshape(a,[batch_size,1])
    b=tf.reshape(b,[batch_size,1])
    params = tf.concat([a, b, -b, a, gap_mean], 1,name=output)   # shape: 64 * 6
    
    return params

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
    return pair_feature
