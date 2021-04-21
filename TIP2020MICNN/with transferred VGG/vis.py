import numpy as np
import cv2
import os
import matplotlib as m
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import linecache

class_name_abbr = ['Airp', 'Bare', 'Base', 'Beach', 'Bridge', 'Center', 'Church', 'Commer',
                   'Dense R', 'Desert', 'Farm', 'Forest', 'Indus', 'Meadow', 'Medium R',
                   'Mount', 'Park', 'Parking', 'Play', 'Pond', 'Port', 'Rail Sta', 'Resort', 'River',
                   'School', 'Sparse R', 'Square', 'Stadium', 'Storage', 'Viaduct']

class_sample_num = [360, 310, 220, 400, 360, 260, 240, 350, 410, 300, 370, 250, 390, 280, 290, 340, 350, 390, 370, 420,
                    380, 260, 290, 410, 300, 300, 330, 290, 360, 420]


def img_topic_vis(prob, classMap, class_num):
    topic_num = prob.shape[-1]
    local_topics = np.argmax(prob, axis=3)
    classMap = np.squeeze(classMap)
    test_file = 'test.txt'
    start_index = sum(class_sample_num[:class_num]) // 2
    for i in range(class_sample_num[class_num] // 2):
        path = linecache.getline(test_file, start_index + i+1)
        print(path)
        img = cv2.imread(path[:-1])  # 去除换行符
        img = img[:, :, ::-1]  # bgr to rgb
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(131)
        ax1.set_aspect('auto')
        ax1.imshow(img)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(132)
        ax2.set_aspect('auto')
        im1 = ax2.imshow(local_topics[start_index + i, :, :], vmin=0, vmax=topic_num, cmap='hot')
        ax2.set_xticks([])
        ax2.set_yticks([])
        # plt.colorbar(im1)

        ax3 = fig.add_subplot(133)
        ax3.set_aspect('auto')
        im2 = ax3.imshow(classMap[start_index + i, :, :], cmap='jet')
        ax3.set_xticks([])
        ax3.set_yticks([])
        # plt.colorbar(im2)

        plt.show()


def topic_distribution_vis(pzd):
    topic_num = pzd.shape[-1]

    def topic_hist(pzd, topic_num):
        local_topics = np.reshape(pzd, (-1, 196, topic_num))
        local_topics = np.argmax(local_topics, axis=2)

        local_counts = []
        for line in local_topics:
            local_count = np.bincount(line, minlength=topic_num)
            local_counts.append(local_count)
        counts = np.vstack(local_counts)

        return counts

    hist = topic_hist(pzd, topic_num=topic_num)
    hist = np.float32(hist) / 196
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(hist, cmap='hot')
    ax.set_aspect('auto')
    plt.xlabel('classes')
    plt.ylabel('images')
    plt.yticks([sum(class_sample_num[:i]) for i in range(30)], class_name_abbr, size='small')
    plt.xticks(list(range(0, topic_num, 1)), class_name_abbr, size='small')
    ax.set_xticklabels(class_name_abbr, rotation=45, ha="center")
    # ax.tick_params(axis='both', which='major', labelsize=8)
    # plt.axis('off')
    plt.colorbar(im)
    plt.show()

class_name = ['Airp', 'Bare', 'Base', 'Beach', 'Bridge', 'Center', 'Church', 'Commer',
                   'Dense R', 'Desert', 'Farm', 'Forest', 'Indus', 'Meadow', 'Medium R',
                   'Mount', 'Park', 'Parking', 'Play', 'Pond', 'Port', 'Rail Sta', 'Resort', 'River',
                   'School', 'Sparse R', 'Square', 'Stadium', 'Storage', 'Viaduct']
import cv2
def instance_label_vis(instance_label, attention_map, class_num):
    instance_label = np.vstack(instance_label)
    attention_map = np.vstack(attention_map)

    local_topics = np.argmax(instance_label, axis=3)
    attention_maps = np.squeeze(attention_map)
    test_file = 'D:/Dataset/AID_dataset/2-0/test.txt'
    start_index = sum(class_sample_num[:class_num]) // 2

    for i in range(class_sample_num[class_num] // 2):
        path = linecache.getline(test_file, start_index + i + 1)
        print(path)
        img = cv2.imread(path[:-1])  # 去除换行符
        img = cv2.resize(img, dsize=(256, 256))
        img = img[:, :, ::-1]

        labels = []
        label_class = []
        for class_num in range(45):
            label = (local_topics[start_index+i, :, :] == class_num).astype(np.uint8)
            if np.sum(label) > 20:
                label = cv2.resize(label, dsize=(256, 256), interpolation=0)
                labels.append(label)
                label_class.append(class_name[class_num])
        num = len(labels)
        print(num)
        fig = plt.figure(figsize=(2*(num+2), 2))
        ax = fig.add_subplot(1, num+2, 1)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for n in range(num):
            ax = fig.add_subplot(1, num+2, n+2)
            ax.imshow(img)
            ax.imshow(labels[n], 'hot', interpolation='none', alpha=0.5)
            ax.set_title(label_class[n],size=10)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(1, num+2, num+2)
        ax.imshow(img)
        attention_map = cv2.resize(attention_maps[start_index+i, :, :], dsize=(256, 256), interpolation=0)
        ax.imshow(attention_map, cmap='hot', interpolation='none', alpha=0.5)
        ax.set_title('attention', size=10)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.show()

def merge_joblib():
    prob = joblib.load('prob_features_21_train')
    attention_maps = joblib.load('attention_maps_21_train')
    attention_scores = joblib.load('attention_scores_train')
    prob1 = joblib.load('prob_features_21_test')
    attention_maps1 = joblib.load('attention_maps_21_test')
    attention_scores1 = joblib.load('attention_scores_test')

    for n in range(100):
        prob[n] = np.vstack((prob[n], prob1[n]))
        attention_maps[n] = np.vstack((attention_maps[n], attention_maps1[n]))
        attention_scores[n] = np.vstack((attention_scores[n], attention_scores1[n]))
    return prob, attention_maps, attention_scores


# def path_gen(dataset_path, ext=".jpg"):
#     class_names = [f for f in os.listdir(dataset_path) if not f.startswith('.')]
#     list.sort(class_names)
#     print(class_names)
#     train_file = open("train.txt", "w")
#     test_file = open("test.txt", "w")
#
#     for index, name in enumerate(class_names):
#         print(name)
#         directory = os.path.join(dataset_path, name)
#         class_image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)]
#         list.sort(class_image_paths)
#
#         for i, img_path in enumerate(class_image_paths):
#             if (i + 1) % 2 == 0:
#                 test_file.write(img_path+'\n')
#             else:
#                 train_file.write(img_path+'\n')
#     train_file.close()
#     test_file.close()
# path_gen('/Users/lzl/Downloads/AID')

def main():
    # img = joblib.load('imgs')
    attention_map = joblib.load('attention_map')
    instance_score = joblib.load('label_map')

    # label_preds_packed = joblib.load('label_pred_train')
    # img_topic_vis(img, instance_score, attention_map, 16)
    instance_label_vis(instance_score, attention_map, 6)

if __name__ == '__main__':
    main()
# prob, attention_maps, attention_scores = merge_joblib()
# prob_tor = np.vstack(prob)
# # topic_distribution_vis(prob_tor)
# #
# local_topics = np.argmax(prob_tor, axis=3)
# attention_scores_tor = np.vstack(np.squeeze(attention_scores))
# topic_att_score = []
# topic_att_score_median_ori = []
# for i in range(30):
#     # plot_num = i + 1
#     # plt.subplot(6, 4, plot_num)
#     # topic_att_score = attention_scores_tor[local_topics == i]
#     # x = plt.hist(topic_att_score, alpha=0.7, label=class_name[i])
#     ast = attention_scores_tor[local_topics == i]
#     astm = np.median(ast, axis=0)
#     topic_att_score.append(ast)
#     topic_att_score_median_ori.append(astm)
# #
# a = [np.exp(m) for m in topic_att_score]
# zipped = zip(class_name_abbr, topic_att_score, topic_att_score_median_ori)
# zipped = sorted(zipped, key=lambda x: -x[2])
# class_name, topic_att_score, topic_att_score_median = zip(*zipped)
#
#
# # plt.xlabel('attention score', size='small')
# # plt.boxplot(topic_att_score, labels=class_name, vert=False, showfliers=False)
# # plt.tick_params(axis='both', which='major', labelsize='small')
# # plt.show()
#
# def compute_squared_EDM_method(X):
#     # determin dimensions of data matrix
#     n = len(X)
#     # initialize squared EDM D
#     D = np.zeros([n, n])
#     # iterate over upper triangle of D
#     for i in range(n):
#         for j in range(i + 1, n):
#             D[i, j] = abs(X[i] - X[j])
#             D[j, i] = D[i, j]  # *1
#     return D
#
#
# def get_conf_mat():
#     from sklearn.metrics import confusion_matrix
#     label_preds_packed = joblib.load('label_pred')
#     conf_mat = confusion_matrix(label_preds_packed[:, 0], label_preds_packed[:, 1])
#     normed_cm = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)
#     return normed_cm
#
#
# Dis = np.exp(compute_squared_EDM_method(topic_att_score_median_ori))
# conf_mat = get_conf_mat()
#
# Dis = Dis[conf_mat < 0.3]
# conf_mat = conf_mat[conf_mat < 0.3]
# plt.scatter(Dis, conf_mat)
# plt.ylabel('confusion')
# plt.xlabel('relative semantic priority')
# plt.show()
