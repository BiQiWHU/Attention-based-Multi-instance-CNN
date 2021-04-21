import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

class_name_abbr = ['Airp', 'Bare', 'Base', 'Beach', 'Bridge', 'Center', 'Church', 'Commer',
                   'Dense R', 'Desert', 'Farm', 'Forest', 'Indus', 'Meadow', 'Medium R',
                   'Mount', 'Park', 'Parking', 'Play', 'Pond', 'Port', 'Rail Sta', 'Resort', 'River',
                   'School', 'Sparse R', 'Square', 'Stadium', 'Storage', 'Viaduct']

class_sample_num = [360, 310, 220, 400, 360, 260, 240, 350, 410, 300, 370, 250, 390, 280, 290, 340, 350, 390, 370, 420,
                    380, 260, 290, 410, 300, 300, 330, 290, 360, 420]


def img_topic_vis(imgs, prob, attention_map, class_num):
    topic_num = prob.shape[-1]
    local_topics = np.argmax(prob, axis=3)
    attention_map = np.squeeze(attention_map)

    start_index = sum(class_sample_num[:class_num]) // 2

    for i in range(start_index, start_index + class_sample_num[class_num] // 2):
        img = imgs[i, :, :, ::-1]  # bgr to rgb
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(131)
        ax1.set_aspect('auto')
        ax1.imshow(img)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(132)
        ax2.set_aspect('auto')
        ax2.imshow(local_topics[i, :, :], vmin=0, vmax=topic_num, cmap='hot')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = fig.add_subplot(133)
        ax3.set_aspect('auto')
        ax3.imshow(attention_map[i, :, :], cmap='jet')
        ax3.set_xticks([])
        ax3.set_yticks([])

        plt.show()


def img_topic_vis_self(imgs, prob, class_num):
    topic_num = prob.shape[-1]
    local_topics = np.argmax(prob, axis=3)
    attention_map = np.max(prob, axis=3)
    start_index = sum(class_sample_num[:class_num]) // 2

    for i in range(start_index, start_index + class_sample_num[class_num] // 2):
        img = imgs[i, :, :, ::-1]  # bgr to rgb
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(131)
        ax1.set_aspect('auto')
        ax1.imshow(img)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(132)
        ax2.set_aspect('auto')
        ax2.imshow(local_topics[i, :, :], vmin=0, vmax=topic_num, cmap='hot')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = fig.add_subplot(133)
        ax3.set_aspect('auto')
        ax3.imshow(attention_map[i, :, :], cmap='jet')
        ax3.set_xticks([])
        ax3.set_yticks([])

        plt.show()


def class_distribution_vis(instance_score):
    class_num = instance_score.shape[-1]

    def topic_hist(pzd, class_num):
        local_topics = np.reshape(pzd, (-1, 14 * 14, class_num))
        local_topics = np.argmax(local_topics, axis=2)
        local_counts = []
        for line in local_topics:
            local_count = np.bincount(line, minlength=class_num)
            local_counts.append(local_count)
        counts = np.vstack(local_counts)
        return counts

    hist = topic_hist(instance_score, class_num=class_num)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(hist, cmap='hot')
    ax.set_aspect('auto')
    plt.xlabel('Topic ID')
    plt.ylabel('Image ID')
    plt.yticks(list(range(0, hist.shape[0], 80)))
    plt.xticks(list(range(0, class_num, 1)))
    plt.colorbar(im)
    plt.show()


def priority(instance_score, attention_score):
    local_topics = np.argmax(instance_score, axis=3)
    attention_score = np.vstack(np.squeeze(attention_score))
    topic_att_score = []
    topic_att_score_median_ori = []
    for i in range(21):
        ast = attention_score[local_topics == i]
        astm = np.median(ast, axis=0)
        topic_att_score.append(ast)
        topic_att_score_median_ori.append(astm)

    zipped = zip(class_name_abbr, topic_att_score, topic_att_score_median_ori)
    zipped = sorted(zipped, key=lambda x: -x[2])
    class_name, topic_att_score, topic_att_score_median = zip(*zipped)

    plt.xlabel('attention score', size='small')
    plt.boxplot(topic_att_score, labels=class_name, vert=False, showfliers=False)
    plt.tick_params(axis='both', which='major', labelsize='small')
    plt.show()


def priority_self(instance_score):
    local_topics = np.argmax(instance_score, axis=3)
    topic_att_score = []
    topic_att_score_median_ori = []
    for i in range(21):
        ast = np.max(instance_score, axis=3)[local_topics == i]
        astm = np.median(ast, axis=0)
        topic_att_score.append(ast)
        topic_att_score_median_ori.append(astm)

    zipped = zip(class_name_abbr, topic_att_score, topic_att_score_median_ori)
    zipped = sorted(zipped, key=lambda x: -x[2])
    class_name, topic_att_score, topic_att_score_median = zip(*zipped)

    plt.xlabel('attention score', size='small')
    plt.boxplot(topic_att_score, labels=class_name, vert=False, showfliers=False)
    plt.tick_params(axis='both', which='major', labelsize='small')
    plt.show()


def confusion_mat(label_preds_packed):
    conf_mat = confusion_matrix(label_preds_packed[:, 0], label_preds_packed[:, 1])
    normed_cm = conf_mat / np.sum(conf_mat, axis=1, keepdims=True)
    print(normed_cm.diagonal())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(normed_cm, vmin=0.0, vmax=1.0,  cmap='binary')
    ax.set_aspect('auto')
    plt.yticks(list(range(30)), class_name_abbr, size='small')
    plt.xticks(list(range(30)), class_name_abbr, size='small')
    ax.set_xticklabels(class_name_abbr, rotation=45, ha="center")
    for i in range(30):
        for j in range(30):
            c = normed_cm[j, i]
            if c > 0.001:
                if i == j:
                    ax.text(i, j, str(c)[:5], va='center', ha='center', size=6, color='white')
                else:
                    ax.text(i, j, str(c)[:5], va='center', ha='center', size=6, color='black')
    plt.colorbar(im)
    plt.show()

def main():
    img = joblib.load('img')
    instance_score = joblib.load('instance_score')
    attention_score = joblib.load('attention_score')
    attention_map = joblib.load('attention_map')
    label_preds_packed = joblib.load('label_pred_train')

    img = np.vstack(img)
    instance_score = np.vstack(instance_score)
    attention_map = np.vstack(attention_map)

    # img_topic_vis(img, instance_score, attention_map, class_num = 29)
    #class_distribution_vis(instance_score)
    #priority(instance_score, attention_score)
    confusion_mat(label_preds_packed)
    # priority_self(instance_score)
    # img_topic_vis_self(img, instance_score, class_num=6)

if __name__ == '__main__':
    main()