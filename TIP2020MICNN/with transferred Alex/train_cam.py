from network_cam import *
from tfdata import *
import numpy as np


def bag_ce(logits, targets):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy


def l2_reg():
    weights_only = filter(lambda x: x.name.endswith('weights:0'), tf.trainable_variables())
    l2_regularization = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only]))
    return l2_regularization


def accuracy_of_batch(logits, targets):
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


def load_with_skip(data_path, session, skip_layer, scope_name='model_definition'):
    data_dict = np.load(data_path, encoding="bytes").item()
    with tf.variable_scope(scope_name):
        for key in data_dict:
            if key not in skip_layer:
                with tf.variable_scope(key, reuse=True):
                    for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                        get_var = tf.get_variable(subkey).assign(data)
                        session.run(get_var)


def get_variables_to_restore():
    checkpoint_exclude_scopes = ['model_definition/conv9','model_definition/attention_bn','model_definition/final_bn']
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    variables_to_restore = []
    for var in tf.trainable_variables():
        print(var)
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def main(num):
    # Dataset path
    train_tfrecords = 'D:/Dataset/AID_dataset/2-' + str(num) +'/train.tfrecords'
    test_tfrecords = 'D:/Dataset/AID_dataset/2-' + str(num) +'/test.tfrecords'
    alexnet_npy_path = 'D:/ImageNet_Model/bvlc_alexnet.npy'
    # ckpt = 'D:/python/LZL2/AlexMIL-AID-Fusion-Mean/' + '2-' + str(num) + '/checkpoints/my-model.ckpt-' + str(15000)

    save_ckpt_path = '2-' + str(num) + '/checkpoints/my-model.ckpt'
    log_dir = '2-' + str(num) + '/log'

    # Learning params
    learning_rate_ini = 0.0001
    training_iters = 15000
    batch_size = 32

    # Load batch
    train_img, train_label = input_pipeline(train_tfrecords, batch_size)
    test_img, test_label = input_pipeline(test_tfrecords, batch_size, is_train=False)

    # Model
    train_instance_pooling = semantic_alex_net_fusion(train_img, is_training=True)
    test_instance_pooling = semantic_alex_net_fusion(test_img, reuse=True, is_training=False)

    # Loss and optimizer
    bag_loss = bag_ce(train_instance_pooling, train_label)
    l2_regularization = l2_reg()

    loss_sum = bag_loss + 0.0005 * l2_regularization

    global_step = tf.Variable(0, trainable=False)
    global_step_add = global_step.assign_add(1)
    learning_rate = tf.train.exponential_decay(learning_rate_ini, global_step, 5000, 0.1, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_sum)

    # Evaluation
    train_accuracy = accuracy_of_batch(train_instance_pooling, train_label)
    test_accuracy = accuracy_of_batch(test_instance_pooling, test_label)

    # Init
    init = tf.global_variables_initializer()

    # Summary
    tf.summary.scalar('bag_ce_loss', bag_loss)
    tf.summary.scalar("train accuracy", train_accuracy)
    tf.summary.scalar("test accuracy", test_accuracy)

    merged_summary_op = tf.summary.merge_all()

    # Create Saver
    variables_to_restore = get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)

        print('load from alexnet npy pretrained model')
        load_with_skip(alexnet_npy_path, sess, ['fc6', 'fc7', 'fc8'])
        # print('load from ckpt model')
        # restorer.restore(sess, ckpt)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        print('Start training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(training_iters):
            step += 1
            _, _, bag_ce_value = sess.run([global_step_add, train_op, bag_loss])
            print('Generation {}: Bag Loss = {:.5f}'.format(step, bag_ce_value))

            # Display status
            if step % 50 == 0:
                acc1, acc2, summary_str = sess.run([train_accuracy, test_accuracy, merged_summary_op])
                print(' --- Train Accuracy = {:.2f}%.'.format(100. * acc1))
                print(' --- Test Accuracy = {:.2f}%.'.format(100. * acc2))
                summary_writer.add_summary(summary_str, global_step=step)
            if step % 200 == 0:
                saver.save(sess, save_ckpt_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

        print("Finish!")

    print("Reload graph")
    tf.reset_default_graph()


if __name__ == '__main__':
    for i in range(0, 1):
        if i is not 4:
            print(i)
            main(i)
