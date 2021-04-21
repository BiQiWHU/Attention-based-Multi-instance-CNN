from train_cam import accuracy_of_batch
from network_cam import semantic_alex_net
from tfdata import *
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class_name_abbr = ['Airp', 'Bare', 'Base', 'Beach', 'Bridge', 'Center', 'Church', 'Commer',
                   'Dense R', 'Desert', 'Farm', 'Forest', 'Indus', 'Meadow', 'Medium R',
                   'Mount', 'Park', 'Parking', 'Play', 'Pond', 'Port', 'Rail Sta', 'Resort', 'River',
                   'School', 'Sparse R', 'Square', 'Stadium', 'Storage', 'Viaduct']


def get_label_pred(num, ckpt_num, is_save=False):
    test_tfrecords = 'D:/Dataset/AID_dataset/2-' + str(num) + '/test.tfrecords'
    ckpt = '2-' + str(num) + '/checkpoints/my-model.ckpt-' + str(ckpt_num)
    batch_size = 50

    img, label = input_pipeline(test_tfrecords, batch_size, is_shuffle=False, is_train=False)
    instance_pooling, conv6, attention = semantic_alex_net(img, is_training=False)

    accuracy = accuracy_of_batch(instance_pooling, label)
    pred = tf.cast(tf.argmax(instance_pooling, 1), tf.int32)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        label_preds = []
        label_map = []
        attention_map = []
        imgs = []
        acc_sum = 0
        for i in range(100):
            acc, label_, pred_, img_, conv6_, attention_ = sess.run([accuracy, label, pred, img, conv6, attention])
            x = np.hstack([label_[:, np.newaxis], pred_[:, np.newaxis]])
            label_preds.append(x)
            label_map.append(conv6_)
            attention_map.append(attention_)
            imgs.append(img_)
            acc_sum += acc
            # print(acc)
        print('mean_acc:', acc_sum / 100)
        label_preds_packed = np.vstack(label_preds)
        if is_save:
            joblib.dump(label_preds_packed, 'label_pred_train')
            joblib.dump(label_map, 'label_map')
            joblib.dump(attention_map, 'attention_map')
            joblib.dump(imgs, 'imgs')

        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()


# for i in range(0, 5):
#     for j in range(6000, 5000, -200):
#         print(i, j)
#         get_label_pred(i, j, False)
get_label_pred(0, 15000, True)
