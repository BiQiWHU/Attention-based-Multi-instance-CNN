# coding: utf-8

# In[1]:
from DenseRS import *
from tfdata import *
import numpy as np
from DenseRS import accuracy_of_batch
import os
import re
#import xlwt

# In[2]:
import tensorflow as tf

# In[3]:
import time

# In[4]:

# Dataset path
train_tfrecords = 'train.tfrecords'
test_tfrecords = 'test.tfrecords'
batch_size = 20

# In[5]:

img, label = input_pipeline(test_tfrecords, batch_size, is_shuffle=False, is_train=False)
with tf.variable_scope('model_definition'):
    prediction = MIDCCNN3T23(img, is_training=False, keep_prob=1)
accuracy = accuracy_of_batch(prediction, label)

# In[6]:

saver = tf.train.Saver()

# In[7]:
f = "checkpoint/"

# 从checkpoint文件夹中提取出模型名称
fs = os.listdir(f)
fs1 = []
for f1 in fs:
    (f1name, f1extension) = os.path.splitext(f1)
    fs1.append(f1name)

fs1 = list(set(fs1))
fs1=sorted(fs1,key = lambda i:int(re.search(r'(\d+)',i).group()))
#print(fs1)

## creat txt file
file1 = open("name.txt",'w')
file2 = open("acc.txt",'w')
file3 = open("time.txt",'w')

row = 0

for f1 in fs1:  # 对每个模型进行操作
    with tf.Session() as sess:

        print(f+f1)

        saver.restore(sess, f + f1)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        acc2 = 0

        start = time.clock()

        for i in range(250):
            acc = sess.run(accuracy)
            # print(acc)
            acc2 += acc

        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
        print('OA={:.2f}%'.format(acc2 * 100 / 250))

        with open('name.txt', 'a') as file1:
            file1.write(f1+'\n')

        with open('acc.txt', 'a') as file2:
            file2.write("{:.2f}%".format(acc2 * 100 / 250)+'\n')

        with open('time.txt', 'a') as file3:
            file3.write(str(elapsed)+'\n')

        ### write into txt
        #file1.write(f1+'\n')
        #file2.write("{:.2f}%".format(acc2 * 100 / 250)+'\n')
        #file3.write(str(elapsed)+'\n')

        coord.request_stop()
        coord.join(threads)
        row = row + 1

## save txt
file1.close()
file2.close()
file3.close()

print('finish!')




