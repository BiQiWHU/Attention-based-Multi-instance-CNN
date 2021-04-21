# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 22:23:11 2018
15 versions of DenseNet for  
@author: Qi Bi
"""
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
from tfdata import *
import numpy as np
import tensorflow as tf


# In[2]:


def weight_variable(shape, name):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name='weights',
                                  shape=shape,
                                  trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        REGULARIZATION_RATE=0.000005
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        tf.add_to_collection('losses', regularizer(weights))
        return weights


# In[3]:


def bias_variable(shape, name):
    with tf.variable_scope(name) as scope:
        biases = tf.get_variable(name='biases',
                                 shape=shape,
                                 trainable=True,
                                 initializer=tf.constant_initializer(0.01))

        return biases


# In[4]:


def conv2d(input, in_feature_dim, out_feature_dim, kernel_size, stride, with_bias=True, name=None):
    W = weight_variable([kernel_size, kernel_size, in_feature_dim, out_feature_dim], name=name)
    conv = tf.nn.conv2d(input, W, [1, stride, stride, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_feature_dim], name=name)
    return conv


# In[5]:


def batch_activ_conv(current, in_feature_dim, out_feature_dim, kernel_size, is_training, keep_prob, name):
    with tf.variable_scope(name) as scope:
        # current = tf.contrib.layers.batch_norm(current, decay=0.9, scale=True, is_training=False,
        #                                       updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name)
        ### test1 BN=0.9  test=0.99
        if is_training:
            current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name)
        else:
            current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training, scope=name)
        current = tf.nn.relu(current)
        current = conv2d(current, in_feature_dim, out_feature_dim, kernel_size, stride=1, name=name)
        current = tf.nn.dropout(current, keep_prob)
        return current


# In[6]:


### growth: feature maps that each layer preduce, equals to the number of filters
def block(input, layers, in_feature_dim, growth, is_training, keep_prob, name):
    current = input
    sum_feature_dim = in_feature_dim
    for id in range(layers):
        ### for the orginal Dense, please remove the first code on 1*1 filters
        tmp = batch_activ_conv(current, sum_feature_dim, growth, 1, is_training, keep_prob,
                               name=name + '/conv1_layer_%d' % id)

        tmp = batch_activ_conv(tmp, growth, growth, 3, is_training, keep_prob,
                               name=name + '/conv3_layer_%d' % id)
        current = tf.concat((current, tmp), 3)
        sum_feature_dim += growth
    return current, sum_feature_dim


# In[7]:


def avg_pool(input, s, stride):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, stride, stride, 1], 'SAME')


# In[8]:


def max_pool(input, s, stride):
    return tf.nn.max_pool(input, [1, s, s, 1], [1, stride, stride, 1], 'SAME')


# In[9]:


def loss(logits, targets):
    # Get rid of extra dimensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Calculate cross entropy from logits and targets
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # Take the average loss across batch size
    #cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy') + tf.add_n(tf.get_collection('losses'))
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


# In[10]:


# Train step
def train(loss_value, model_learning_rate):
    # Create optimizer
    # my_optimizer = tf.train.MomentumOptimizer(model_learning_rate, momentum=0.9)

    my_optimizer = tf.train.AdamOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return train_step


# In[11]:


# Accuracy function
def accuracy_of_batch(logits, targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


# In[12]:


def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding="bytes").item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    get_var = tf.get_variable(subkey).assign(data)
                    session.run(get_var)


# In[13]:


def fc(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:
        Wfc = weight_variable([num_in, num_out], name=name)
        bfc = bias_variable([num_out], name=name)

        tf.summary.histogram(name + "/weights", Wfc)
        tf.summary.histogram(name + "/biases", bfc)

        act = tf.nn.xw_plus_b(x, Wfc, bfc, name=name + '/op')

        return act


# In[14]:


def DenseNet121B(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 6, 64, 32, is_training, keep_prob, name='denseblock1')

    ### transition layer1    if we need compression   the third paramter of BAC  plus 0.5
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 12, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 24, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### transition layer3
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### denseblock4
    current, featurenumber = block(current, 16, featurenumber, 32, is_training, keep_prob, name='denseblock4')

    ### transition layer4    wrong, this does not exist
    # current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer4')
    # current = avg_pool(current, 2, 2)

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output

def DenseNet3T11(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 1, 64, 32, is_training, keep_prob, name='denseblock1')

    ### transition layer1    if we need compression   the third paramter of BAC  plus 0.5
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 1, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 1, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### transition layer3
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### this is an addition layer for 4T12
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer4')

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output

def DenseNet3T13(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 1, 64, 32, is_training, keep_prob, name='denseblock1')

    ### transition layer1    if we need compression   the third paramter of BAC  plus 0.5
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 2, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 1, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### transition layer3
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### this is an addition layer for 4T14
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer4')

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output

def DenseNet3T15(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 1, 64, 32, is_training, keep_prob, name='denseblock1')

    ### transition layer1    if we need compression   the third paramter of BAC  plus 0.5
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 2, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 2, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### transition layer3
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output

def DenseNet3T17(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 2, 64, 32, is_training, keep_prob, name='denseblock1')

    ### transition layer1    if we need compression   the third paramter of BAC  plus 0.5
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 2, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 2, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### transition layer3
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### this is an addition layer for 4T14
    #current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer4')

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output

def DenseNet3T19(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 2, 64, 32, is_training, keep_prob, name='denseblock1')

    ### transition layer1    if we need compression   the third paramter of BAC  plus 0.5
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 2, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### transition layer3
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output

def DenseNet3T21(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 2, 64, 32, is_training, keep_prob, name='denseblock1')

    ### transition layer1    if we need compression   the third paramter of BAC  plus 0.5
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### transition layer3
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output

def DenseNet3T23(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 3, 64, 32, is_training, keep_prob, name='denseblock1')

    ### transition layer1    if we need compression   the third paramter of BAC  plus 0.5
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### transition layer3
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### this is an addition layer for 4T14
    #current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer4')

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output

#### add the multiple instance learning here

def instance_loss(logits, targets):
    # y_ = tf.one_hot(tf.cast(targets, tf.int32), depth=21)
    # cross_entropy = tf.reduce_mean(
    #     -tf.reduce_sum(tf.multiply(y_, tf.log(tf.clip_by_value(instance_prob, 1e-16, 1.0))), axis=-1))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy


def l2_reg():
    weights_only = filter(lambda x: x.name.endswith('weights:0'), tf.trainable_variables())
    l2_regularization = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only]))
    return l2_regularization



def get_attention_map(conv6):
    with tf.variable_scope('attention_model') as scope:
        # get the height and width of the input tensor
        _, height, width, _ = conv6.get_shape().as_list()
        attention_net = conv2d(conv6, 30, 64, 1, 1, with_bias=False, name='conv7')

        attention_net = tf.tanh(attention_net)
        #attention_net = tf.nn.relu(attention_net)
        
        attention_net = conv2d(attention_net, 64, 1, 1, 1, with_bias=False, name='conv8')
        
        attention = tf.reshape(attention_net, [-1, height * width, 1])
        attention = tf.reshape(tf.nn.softmax(attention, 1, name='compatibility'), [-1, height, width, 1])

        tf.summary.histogram("attentive_act", attention_net)
        tf.summary.histogram("attention", attention)
        
#        ### code from Zhili Li
#        _, height, width, _ = conv6.get_shape().as_list()
#
#        attention_net = conv(conv6, 1, 1, 64, 1, 1, name='conv7', relu=False, plus_bias=False)
#        attention_net = tf.tanh(attention_net)
#        attention_net = conv(attention_net, 1, 1, 1, 1, 1, name='conv8', relu=False, plus_bias=False)
#
#        attention = tf.reshape(attention_net, [-1, height * width, 1])
#        attention = tf.reshape(tf.nn.softmax(attention, 1, name='compatibility'), [-1, height, width, 1])
#
#        tf.summary.histogram("attentive_act", attention_net)
#        tf.summary.histogram("attention", attention)

        return attention

def get_attention_cs(conv6):
    with tf.variable_scope('attention_model') as scope:
            # get the height and width of the input tensor
        _, height, width, _ = conv6.get_shape().as_list()
        attention_net = conv2d(conv6, 30, 64, 1, 1, with_bias=False, name='conv7')

            ### channel wise
            # GAP
        attention_c = avg_pool(attention_net, 7, 7)
            # fc1  channel=channel/16
            # attention_c = fc(attention_c, 64, 4, name='att_fc1')  # relu
        attention_c = conv2d(attention_c, 64, 16, 1, 1, with_bias=False, name='att_fc1')
            # relu
        attention_c = tf.nn.relu(attention_c)
            # fc2 channel back
            # attention_c = fc(attention_c, 4, 64, name='att_fc2')
        attention_c = conv2d(attention_c, 16, 64, 1, 1, with_bias=False, name='att_fc2')
            # sigmoid
        attention_c = tf.nn.sigmoid(attention_c)
            # reshape
        attention_c = tf.reshape(attention_c, [-1, 1, 1, 64])
            # scale
        attention_net = attention_net * attention_c

        attention_net = tf.tanh(attention_net)

        attention_net = conv2d(attention_net, 64, 1, 1, 1, with_bias=False, name='conv8')

        attention = tf.reshape(attention_net, [-1, height * width, 1])
        attention = tf.reshape(tf.nn.softmax(attention, 1, name='compatibility'), [-1, height, width, 1])

        tf.summary.histogram("attentive_act", attention_net)
        tf.summary.histogram("attention", attention)

        return attention


###   MI-DCCNN
def MIDCCNN3T23(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 3, 64, 32, is_training, keep_prob, name='denseblock1')

    ### transition layer1    if we need compression   the third paramter of BAC  plus 0.5
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### transition layer2
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### transition layer3
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### this is an addition layer for 4T14
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer4')

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    
    #current = tf.nn.relu(current)
    #current = avg_pool(current, 7, 7)
    
    current = conv2d(current, featurenumber, 30, 1, 1, with_bias=False, name='conv6')
    #attention_map = get_attention_map(current)
    attention_map = get_attention_cs(current)
    #attention_map = avg_pool(current, 7, 1)
    instance_pooling = tf.reduce_sum(tf.multiply(current, attention_map), [1, 2])
    
    return instance_pooling

#    current = avg_pool(current, 7, 7)
#    final_dim = featurenumber
#    current = tf.reshape(current, [-1, final_dim])
#    output = fc(current, final_dim, 30, name='fc')
#    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
#    return output

def attention(current,featurenumber,is_training, name):
    # get the height and width of the input tensor
    _, height, width, _ = current.get_shape().as_list()

    #### solution1
    #attention_net = conv2d(current, featurenumber, 64, 1, 1, with_bias=False, name=name+'conv1')

    #attention_net = tf.tanh(attention_net)
    # attention_net = tf.nn.relu(attention_net)

    #attention_net = conv2d(attention_net, 64, 1, 1, 1, with_bias=False, name=name+'conv2')

    ### solution2

    if is_training:
        current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                               updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name)
    else:
        current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training, scope=name)
    current = tf.nn.relu(current)

    attention_net = conv2d(current, featurenumber, 1, 1, 1, with_bias=False, name=name + 'conv1')
    #attention_net = tf.nn.relu(attention_net)

    attention = tf.reshape(attention_net, [-1, height * width, 1])
    attention = tf.reshape(tf.nn.softmax(attention, 1, name=name+'compatibility'), [-1, height, width, 1])

    return attention


###   A3DCCNN
def A3DCCNN3T23(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 3, 64, 32, is_training, keep_prob, name='denseblock1')

    ### attention block1
    attention1 = attention(current,featurenumber,is_training, name='attentionblock1')
    #current = tf.reduce_sum(tf.multiply(current, attention1), [1, 2])
    attention1 = tf.multiply(current, attention1)
    current=current+attention1
    current = tf.nn.dropout(current, keep_prob)
    #current = tf.nn.relu(current)
    ## downsample1
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### attention block2
    attention2 = attention(current, featurenumber, is_training, name='attentionblock2')
    #current = tf.reduce_sum(tf.multiply(current, attention2), [1, 2])
    attention2 = tf.multiply(current, attention2)
    current = current + attention2
    current = tf.nn.dropout(current, keep_prob)
    #current = tf.nn.relu(current)
    ## downsample2
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### attention block3
    attention3 = attention(current, featurenumber, is_training, name='attentionblock3')
    #current = tf.reduce_sum(tf.multiply(current, attention3), [1, 2])
    attention3 = tf.multiply(current, attention3)
    current=current+attention3
    current = tf.nn.dropout(current, keep_prob)
    #current = tf.nn.relu(current)
    ### downsample3
    current = avg_pool(current, 2, 2)

    ### this is an addition layer for 4T14
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer4')

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同
    return output

def attentiondown(current, featurenumber, is_training, name):
    # get the height and width of the input tensor
    _, height, width, _ = current.get_shape().as_list()

    if is_training:
        current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                               updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name)
    else:
        current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training, scope=name)
    current = tf.nn.relu(current)

    attention_net = conv2d(current, featurenumber, 1, 1, 1, with_bias=False, name=name + 'conv1')
    # attention_net = tf.nn.relu(attention_net)

    attention = tf.reshape(attention_net, [-1, height * width, 1])
    attention = tf.reshape(tf.nn.softmax(attention, 1, name=name + 'compatibility'), [-1, height, width, 1])

    return attention


###   ADDNet
def ADDNet(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1 inout 56*56
    current, featurenumber = block(current, 3, 64, 32, is_training, keep_prob, name='denseblock1')

    ## downsample1   28*28
    #current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer1')
    current = avg_pool(current, 2, 2)

    ### attention block1
    attention1 = attentiondown(current, featurenumber, is_training, name='attentionblock1')

    attention1 = tf.multiply(current, attention1)
    current = current + attention1
    current = tf.nn.dropout(current, keep_prob)
    # current = tf.nn.relu(current)

    ## DAG1  28*28->1*1
    attention1 = avg_pool(attention1, 28, 28)
    attention1 = tf.reshape(attention1, [-1, featurenumber])
    output1 = fc(attention1, featurenumber, 30, name='fc_block1')
    output1 = tf.nn.softmax(output1)

    ### denseblock2  28*28
    #current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock2')
    ## downsample2  28*28->14*14
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer2')
    current = avg_pool(current, 2, 2)

    ### attention block2
    attention2 = attentiondown(current, featurenumber, is_training, name='attentionblock2')
    attention2 = tf.multiply(current, attention2)
    current = current + attention2
    current = tf.nn.dropout(current, keep_prob)
    # current = tf.nn.relu(current)

    ## DAG2  14*14->1*1
    attention2 = avg_pool(attention2, 14, 14)
    attention2 = tf.reshape(attention2, [-1, featurenumber])
    output2 = fc(attention2, featurenumber, 30, name='fc_block2')
    output2 = tf.nn.softmax(output2)

    ### denseblock3
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock3')
    ### downsample3
    #current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer3')
    current = avg_pool(current, 2, 2)

    ### attention block3
    attention3 = attentiondown(current, featurenumber, is_training, name='attentionblock3')
    attention3 = tf.multiply(current, attention3)
    current = current + attention3
    current = tf.nn.dropout(attention3, keep_prob)
    # current = tf.nn.relu(current)


    ### this is an addition layer for 4T14
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer4')

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])

    output = fc(current, final_dim, 30, name='fc')
    output = tf.nn.softmax(output)

    output = output1 + 2*output2 + 4*output

    return output

###   DCANet-3T23
def DCANet3T23(xs, is_training, keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### initial conv and pool   for ImageNet,k=32
    current = conv2d(current, 3, 64, 7, 2, name='preprocessing')
    current = max_pool(current, 3, 2)

    #### In Each Dense Block, 1*1 and 3*3 convs are implemented one after one
    ### denseblock1
    current, featurenumber = block(current, 3, 64, 32, is_training, keep_prob, name='denseblock1')

    ### attention block1
    attention1 = attention(current,featurenumber,is_training, name='attentionblock1')
    #current = tf.reduce_sum(tf.multiply(current, attention1), [1, 2])

    attention1 = tf.multiply(current, attention1)
    #current=current+attention1
    current = tf.concat((current, attention1), 3)
    _, _, _, featurenumber1 = current.get_shape().as_list()

    ### added
    current = conv2d(current, featurenumber1, featurenumber, 1, 1, with_bias=True, name='compress1')

    current = tf.nn.dropout(current, keep_prob)
    current = tf.nn.relu(current)

    ## downsample1
    current = avg_pool(current, 2, 2)

    ### denseblock2
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock2')

    ### attention block2
    attention2 = attention(current, featurenumber, is_training, name='attentionblock2')
    #current = tf.reduce_sum(tf.multiply(current, attention2), [1, 2])
    attention2 = tf.multiply(current, attention2)

    #current = current + attention2
    current = tf.concat((current, attention2), 3)
    _, _, _, featurenumber2 = current.get_shape().as_list()

    ### added
    current = conv2d(current, featurenumber2, featurenumber, 1, 1, with_bias=True, name='compress2')

    current = tf.nn.dropout(current, keep_prob)
    current = tf.nn.relu(current)
    ## downsample2
    current = avg_pool(current, 2, 2)

    ### denseblock3
    current, featurenumber = block(current, 3, featurenumber, 32, is_training, keep_prob, name='denseblock3')

    ### attention block3
    attention3 = attention(current, featurenumber, is_training, name='attentionblock3')
    #current = tf.reduce_sum(tf.multiply(current, attention3), [1, 2])
    attention3 = tf.multiply(current, attention3)

    #current=current+attention3
    current = tf.concat((current, attention3), 3)
    _, _, _, featurenumber3 = current.get_shape().as_list()

    ### added
    current = conv2d(current, featurenumber3, featurenumber, 1, 1, with_bias=True, name='compress3')

    current = tf.nn.dropout(current, keep_prob)
    current = tf.nn.relu(current)
    ### downsample3
    current = avg_pool(current, 2, 2)

    ### this is an addition layer for 4T14
    current = batch_activ_conv(current, featurenumber, featurenumber, 1, is_training, keep_prob, name='trans_layer4')

    ### note there is a parameter for BN
    current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                           updates_collections=tf.GraphKeys.UPDATE_OPS, scope='out_of_dense')
    current = tf.nn.relu(current)
    current = avg_pool(current, 7, 7)
    final_dim = featurenumber
    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 30, name='fc')
    ###   此处21应当根据不同数据集的类别数作出修改，例如UC=21，whugf2=45
    return output

# In[18]:
'''
    DenseNet3T  3Tfor only 3 dense blocks and transition layers
      11,13,15,17,19,21,23 mean 7 solutions with the number of layers 
'''

def main():
    # Dataset path
    train_tfrecords = 'train.tfrecords'
    test_tfrecords = 'test.tfrecords'

    # Learning params  原来imagenet的学习率是0.001
    learning_rate = 0.0001
    training_iters = 750000  # 一个epoch5000次
    batch_size = 20

    # Load batch
    train_img, train_label = input_pipeline(train_tfrecords, batch_size)
    test_img, test_label = input_pipeline(test_tfrecords, batch_size)

    # Model
    with tf.variable_scope('model_definition') as scope:
        train_output = MIDCCNN3T23(train_img, is_training=True, keep_prob=0.8)
        scope.reuse_variables()
        test_output = MIDCCNN3T23(test_img, is_training=False, keep_prob=1)
        
#        train_output = DenseNet3T23(train_img, is_training=True, keep_prob=0.8)
#        scope.reuse_variables()
#        test_output = DenseNet3T23(test_img, is_training=False, keep_prob=1)

    # Loss and optimizer
    loss_op = instance_loss(train_output, train_label)
    #l2_regularization = l2_reg()
    #loss_sum = loss_op + 0.0005 * l2_regularization
    loss_sum = loss_op

    tf.summary.scalar('loss', loss_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = train(loss_sum, learning_rate)
        test_loss_op = instance_loss(test_output, test_label)
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    # Evaluation
    train_accuracy = accuracy_of_batch(train_output, train_label)
    tf.summary.scalar("train_accuracy", train_accuracy)

    test_accuracy = accuracy_of_batch(test_output, test_label)
    tf.summary.scalar("test_accuracy", test_accuracy)

    # Init
    init = tf.global_variables_initializer()

    # Summary
    merged_summary_op = tf.summary.merge_all()

    # Create Saver
    # saver = tf.train.Saver(tf.trainable_variables())
    ### the default saver is tf.train.Saver() However use this leads to mistakes
    # saver = tf.train.Saver()

    ### new solution
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list,max_to_keep=2000)

    # Launch the graph
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)
        #with tf.variable_scope('model_definition'):
        #    load_with_skip('bvlc_alexnet.npy', sess, ['fc'])

        load_ckpt_path = 'checkpoint/my-model.ckpt-356500'
        saver.restore(sess, load_ckpt_path)

        summary_writer = tf.summary.FileWriter('logs', sess.graph)

        print('Start training')
        # coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess)
        for step in range(training_iters):
            step += 1
            # _, loss_value = sess.run([train_op, loss_op])
            # print('Generation {}: Loss = {:.5f}'.format(step, loss_value))
            # print(Wfc1value[1, 1], Wfc2value[1, 1])
            _, loss_value, test_loss_value = sess.run([train_op, loss_op, test_loss_op])
            print('Generation {}: Loss = {:.5f}     Test Loss={:.5f}'.format(step, loss_value, test_loss_value))

            # Display testing status
            if step % 50 == 0:
                acc1 = sess.run(train_accuracy)
                print(' --- Train Accuracy = {:.2f}%.'.format(100. * acc1))
                acc2 = sess.run(test_accuracy)
                print(' --- Test Accuracy = {:.2f}%.'.format(100. * acc2))

            if step % 50 == 0:
                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step % 500 == 0:
                saver.save(sess, 'checkpoint/my-model.ckpt', global_step=step)

        print("Finish Training and validation!")

        # coord.request_stop()
        # coord.join(threads)


# In[19]:


if __name__ == '__main__':
    main()


