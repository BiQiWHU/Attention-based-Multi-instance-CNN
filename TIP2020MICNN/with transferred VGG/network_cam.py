from my_ops import *
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg


def semantic_alex_net(x, reuse=False, is_training=True):
    with tf.variable_scope('vgg_16', reuse=reuse) as scope:
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.repeat(x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5/conv5_3', activation_fn=None)
            conv5 = tf.layers.batch_normalization(net, training=is_training, momentum=0.999)
            conv5 = tf.nn.relu(conv5)

            conv6_0 = conv(conv5, 1, 1, 256, 1, 1, name='conv6_0', bn=True, is_training=is_training)

            conv6_1 = dilated_conv(conv5, 3, 3, 256, 1, name='conv6_1', bn=True, is_training=is_training)
            conv6_3 = dilated_conv(conv5, 3, 3, 256, 3, name='conv6_3', bn=True, is_training=is_training)
            conv6_5 = dilated_conv(conv5, 3, 3, 256, 5, name='conv6_5', bn=True, is_training=is_training)
            conv6_7 = dilated_conv(conv5, 3, 3, 256, 7, name='conv6_7', bn=True, is_training=is_training)

            conv6 = conv6_0 + conv6_1 + conv6_3 + conv6_5 + conv6_7

            conv7 = conv(conv6, 1, 1, 256, 1, 1, name='conv7', bn=True, is_training=is_training)
            conv8 = conv(conv7, 1, 1, 30, 1, 1, name='conv8', relu=False)

            attention_map = spatial_attention(conv7, is_training)
            instance_pooling = tf.reduce_sum(tf.multiply(conv8, attention_map), [1, 2])

            instance_pooling = tf.layers.batch_normalization(instance_pooling, training=is_training, momentum=0.999)

            return instance_pooling, conv8, attention_map


def spatial_attention(conv_feature_map,is_training):
    _, height, width, _ = conv_feature_map.get_shape().as_list()
    attention_net = conv(conv_feature_map, 1, 1, 1, 1, 1, name='conv9', relu=False, plus_bias=False)
    attention_net = tf.nn.sigmoid(attention_net)

    attention = tf.reshape(attention_net, [-1, height * width, 1])
    attention = tf.reshape(tf.nn.softmax(attention, 1), [-1, height, width, 1])
    tf.summary.histogram("attention_score", attention_net)
    tf.summary.histogram("attention_weight", attention)

    return attention