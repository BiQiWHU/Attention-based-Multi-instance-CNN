####  convert AID jpg image into tf format
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 22:45:44 2018

@author: 2009b_000
"""

# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import cv2

# In[2]:


test_ratio = 2
crop_size = 224
scale_size = 600
n_classes = 30


# In[3]:


def get_records(dataset_path, ext=".jpg"):
    ###solution1: 2 datafile  when test do not let make test batch size too large
    writer_train = tf.python_io.TFRecordWriter("train.tfrecords")
    writer_test = tf.python_io.TFRecordWriter("test.tfrecords")

    #### Solution2: 10 small datafile with equal number
    ## train tfrecord
    # writer_train = tf.python_io.TFRecordWriter("train1.tfrecords")
    # writer_train = tf.python_io.TFRecordWriter("train2.tfrecords")
    # writer_train = tf.python_io.TFRecordWriter("train3.tfrecords")
    # writer_train = tf.python_io.TFRecordWriter("train4.tfrecords")
    # writer_train = tf.python_io.TFRecordWriter("train5.tfrecords")

    ## test tfrecord
    # writer_test = tf.python_io.TFRecordWriter("test1.tfrecords")
    # writer_test = tf.python_io.TFRecordWriter("test2.tfrecords")
    # writer_test = tf.python_io.TFRecordWriter("test3.tfrecords")
    # writer_test = tf.python_io.TFRecordWriter("test4.tfrecords")
    # writer_test = tf.python_io.TFRecordWriter("test5.tfrecords")

    class_names = [f for f in os.listdir(dataset_path) if not f.startswith('.')]

    for index, name in enumerate(class_names):
        print(index, ",", name)
        directory = os.path.join(dataset_path, name)
        class_image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)]

        for i, img_path in enumerate(class_image_paths):
            img = cv2.imread(img_path)
            ### for AlexNet  for DenseNet on ImageNet
            # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            ### for DenseNet on CIFAR   DenseNet40 32  for Dense100+ 224
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            # 8 2
            if (i + 1) % test_ratio == 1:
                writer_test.write(example.SerializeToString())
            else:
                writer_train.write(example.SerializeToString())

    writer_train.close()
    writer_test.close()


# In[4]:


def read_and_decode(filename, distort_images=False):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    ### for Alex and Dense on ImageNet ,reshape to 256  For each pixel,in DenseNet we need to normalize it to [0,1]
    # img = tf.reshape(img, [256, 256, 3])
    ### for Dense on CIFAR, reshape to 32
    img = tf.reshape(img, [224, 224, 3]) / 255
    label = tf.cast(features['label'], tf.int32)

    if distort_images:
        # Randomly flip the image horizontally or vertically, change the brightness and contrast
        ##  for AlexNet and DenseNet on ImageNet   random crop
        ##  for DenseNet on CIFAR  do not radom crop
        # img = tf.random_crop(img, [224, 224, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)

    img = tf.image.convert_image_dtype(img, tf.float32)

    return img, label


# In[5]:


def input_pipeline(filename, batch_size, is_shuffle=True, is_train=True):
    example, label = read_and_decode(filename)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    if is_shuffle:
        example_batch, label_batch = tf.train.shuffle_batch([example, label],
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
    else:
        example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)
    return example_batch, label_batch


# In[6]:


if __name__ == '__main__':
    get_records("AID")
    # 0, Airport
    # 1, BareLand
    # 2, BaseballField
    # 3, Beach
    # 4, Bridge
    # 5, Center
    # 6, Church
    # 7, Commercial
    # 8, DenseResidential
    # 9, Desert
    # 10, Farmland
    # 11, Forest
    # 12, Industrial
    # 13, Meadow
    # 14, MediumResidential
    # 15, Mountain
    # 16, Park
    # 17, Parking
    # 18, Playground
    # 19, Pond
    # 20, Pork
    # 21, RailwayStation
    # 22, Resort
    # 23, River
    # 24, School
    # 25, SparseResidential
    # 26, Square
    # 27, Stadium
    # 28, StorageTanks
    # 29, Viaduct

