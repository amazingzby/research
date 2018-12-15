import tensorflow as tf

tfrecordsPath = 'data/train.tfrecords'

imgs,labels = loadTfrecords(tfrecordsPath,32)

