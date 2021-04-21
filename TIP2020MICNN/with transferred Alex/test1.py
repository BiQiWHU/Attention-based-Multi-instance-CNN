from train_cam import accuracy_of_batch
from network_cam import semantic_alex_net_fusion
from tfdata import *
from sklearn.externals import joblib
import numpy as np


def get_label_pred(num, ckpt_num, is_save=False):
    test_tfrecords = 'D:/Dataset/AID_dataset/2-' + str(num) + '/test.tfrecords'
    ckpt = '2-' + str(num) + '/checkpoints/my-model.ckpt-' + str(ckpt_num)
    batch_size = 50

    img, label = input_pipeline(test_tfrecords, batch_size, is_shuffle=False, is_train=False)
    instance_pooling = semantic_alex_net_fusion(img, is_training=False)

    accuracy = accuracy_of_batch(instance_pooling, label)
    pred = tf.cast(tf.argmax(instance_pooling, 1), tf.int32)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        label_preds = []
        acc_sum = 0
        for i in range(100):
            acc, label_, pred_, img_ = sess.run([accuracy, label, pred, img])
            x = np.hstack([label_[:, np.newaxis], pred_[:, np.newaxis]])
            label_preds.append(x)
            acc_sum += acc
        print('mean_acc:', acc_sum / 100)
        label_preds_packed = np.vstack(label_preds)
        if is_save:
            joblib.dump(label_preds_packed, 'label_pred_train')

        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()


for i in range(0, 1):
    for j in range(15000, 14000, -200):
        print(i,j)
        if i is not 4:
            get_label_pred(i, j, False)