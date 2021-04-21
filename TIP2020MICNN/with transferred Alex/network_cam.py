from my_ops import *


def semantic_alex_net_fusion(img_batch, reuse=False, is_training=False):
    with tf.variable_scope('model_definition', reuse=reuse) as scope:
        # Layer 1 (conv-relu-pool-lrn)
        conv1 = conv(img_batch, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5', bn=True, is_training=is_training)

        conv6 = conv(conv5, 3, 3, 512, 1, 1, name='conv6', bn=True, is_training=is_training)
        conv7 = conv(conv6, 3, 3, 512, 1, 1, name='conv7', bn=True, is_training=is_training)

        conv8 = conv(conv7, 1, 1, 30, 1, 1, name='conv8', bn=True, is_training=is_training)

        attention_map = spatial_attention(conv7, is_training)
        instance_pooling = tf.reduce_sum(tf.multiply(conv8, attention_map), [1, 2])

        instance_pooling = tf.layers.batch_normalization(instance_pooling, training=is_training, momentum=0.999, name='final_bn')
    return instance_pooling


def spatial_attention(conv_feature_map,is_training):
    _, height, width, _ = conv_feature_map.get_shape().as_list()
    attention_net = conv(conv_feature_map, 1, 1, 1, 1, 1, name='conv9', relu=False, plus_bias=False)
    attention_net = tf.nn.tanh(attention_net)

    attention = tf.reshape(attention_net, [-1, height * width, 1])
    attention = tf.reshape(tf.nn.softmax(attention, 1), [-1, height, width, 1])
    tf.summary.histogram("attention_score", attention_net)
    tf.summary.histogram("attention_weight", attention)

    return attention