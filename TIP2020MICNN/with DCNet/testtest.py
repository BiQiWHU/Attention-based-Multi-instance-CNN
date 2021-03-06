# coding: utf-8

# In[1]:
from DenseRS import *
from tfdata import *
import numpy as np
from DenseRS import accuracy_of_batch


# In[2]:
import tensorflow as tf

# In[3]:
import cv2
import time

# In[4]:

# Dataset path
train_tfrecords = 'train.tfrecords'
test_tfrecords = 'test.tfrecords'
batch_size = 20

# In[5]:

img, label = input_pipeline(test_tfrecords, batch_size, is_shuffle=False, is_train=False)
with tf.variable_scope('model_definition'):
    prediction = MIDCCNN3T23(img, is_training=False,keep_prob=1)
accuracy = accuracy_of_batch(prediction, label)

# In[6]:

saver = tf.train.Saver()

# In[7]:

with tf.Session() as sess:
    saver.restore(sess, 'checkpoint/my-model.ckpt-174720')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    acc2=0

    start = time.clock()

    for i in range(250):
        acc = sess.run(accuracy)
        print(acc)
        acc2+=acc

    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

    print('OA={:.2f}%'.format(acc2*100/250))
    coord.request_stop()
    coord.join(threads)



