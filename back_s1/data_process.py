import tensorflow as tf
import numpy as np

#npz_file = './data/validation.npz'
tfrecords_path = './data/train.tfrecords'

HEIGHT = 112
WIDTH  = 112

def creatTfrecords(tfrecordsPath):
    with tf.python_io.TFRecordWriter(tfrecordsPath) as writer:
        total = 0
        print('Writing dataset...')
        for i in range(200):
            dataset  = np.load('data/train_processed_'+str(i+1)+'.npz')
            num_imgs = dataset['imgs'].shape[0]

            for idx in range(num_imgs):
                total = total+1
                print('img '+str(total)+' done!')
                img         = np.reshape(dataset['imgs'][idx],(-1))
                gtLandmarks = np.reshape(dataset['gtLandmarks'][idx],(-1))
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                        'image_raw':tf.train.Feature(float_list = tf.train.FloatList(value=img)),
                        'landmark' :tf.train.Feature(float_list = tf.train.FloatList(value=gtLandmarks))
                    }))
                writer.write(example.SerializeToString())
        writer.close()
    print("total imgs:"+str(total))
    print("Done!")

def loadTfrecords(tfrecordsPath,batch_size):
    def parser(record):
        features = tf.parse_single_example(record,
                  features = {
                              'image_raw':tf.FixedLenFeature([HEIGHT*WIDTH], tf.float32),
                              'landmark' :tf.FixedLenFeature([136], tf.float32)
                             })
        img = features['image_raw']
        img = tf.reshape(img, [HEIGHT,WIDTH,1])
        landmark = features['landmark']
        return img,landmark
    dataset = tf.data.TFRecordDataset(tfrecordsPath)
    dataset = dataset.map(parser,num_parallel_calls=32).shuffle(buffer_size=1000).repeat(400).batch(batch_size)
    #dataset = dataset.map(parser,num_parallel_calls=32).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    img_batch,landmark_batch = iterator.get_next()
    return img_batch,landmark_batch

def loadTfrecords_t(tfrecordsPath,batch_size):
    def parser(record):
        features = tf.parse_single_example(record,
                  features = {
                              'image_raw':tf.FixedLenFeature([HEIGHT*WIDTH], tf.float32),
                              'landmark' :tf.FixedLenFeature([136], tf.float32)
                             })
        img = features['image_raw']
        img = tf.reshape(img, [HEIGHT,WIDTH,1])
        landmark = features['landmark']
        return img,landmark
    dataset = tf.data.TFRecordDataset(tfrecordsPath)
    dataset = dataset.map(parser,num_parallel_calls=32).batch(batch_size)
    #dataset = dataset.map(parser,num_parallel_calls=32).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    img_batch,landmark_batch = iterator.get_next()
    return img_batch,landmark_batch

if __name__ == '__main__':
	creatTfrecords(tfrecords_path)

	imgs,landmarks = loadTfrecords(tfrecords_path,32)
	with tf.Session() as sess:
		for i in range(2):
			img_batch,landmarks_batch = sess.run([imgs,landmarks])
			print('image shape:')
			print(img_batch.shape)
		print("Good job!")


