import tensorflow as tf
from  data_process import loadTfrecords
tfrecordsPath = 'data/train.tfrecords'

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/model_iter-0.meta')
    saver.restore(sess,tf.train.latest_checkpoint("model/"))
    graph = tf.get_default_graph()
    x     = graph.get_tensor_by_name('Placeholder:0')
    y_pre = graph.get_tensor_by_name("S2/landmarks_affine:0")
    for i in range(10):
        imgs_batch,landmarks_batch = loadTfrecords(tfrecordsPath,15)
        imgs,labels = sess.run([imgs_batch,landmarks_batch])
        res = sess.run(y_pre,feed_dict={x:imgs})
        print(labels[0])
        print(res[0])
        print('\n')

