import os
import cv2
from sklearn.utils import shuffle


def add_to_trainTxt():
    path = '/home/user/lcx/ccpd_dataset/'
    dir_list = os.listdir(path)
    all_names = []
    for dir in dir_list:
        jpg_names = os.listdir(path + dir)
        for jpg_name in jpg_names:
            all_names.append(path + dir + '/' + jpg_name)
    train_length = int(len(all_names) * 0.7)
    all_names = shuffle(all_names)
    train_names = all_names[:train_length]
    valid_names = all_names[train_length:]
    f = open('train.txt', 'w')
    for train_name in train_names:
        f.write(train_name)
        f.write('\n')
    f.close()
    f = open('valid.txt', 'w')
    for valid_name in valid_names:
        f.write(valid_name)
        f.write('\n')
    f.close()


def generate_label_dir():
    path = '/home/user/lcx/ccpd_dataset'
    dir_list = os.listdir(path)
    for dir in dir_list:
        if not os.path.exists('../data/custom/labels/' + dir):
            os.makedirs('../data/custom/labels/' + dir)


def genenrate_box(img_name):
    img = cv2.imread(img_name)
    # lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]
    iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    # fps = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]
    # leftUp, rightDown = [min([fps[el][0] for el in range(4)]), min([fps[el][1] for el in range(4)])], [
    #     max([fps[el][0] for el in range(4)]), max([fps[el][1] for el in range(4)])]
    [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
    ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
    new_labels = [(leftUp[0] + rightDown[0]) / (2 * ori_w), (leftUp[1] + rightDown[1]) / (2 * ori_h),
                  (rightDown[0] - leftUp[0]) / ori_w, (rightDown[1] - leftUp[1]) / ori_h]
    return new_labels

def generate_label():
    path = '/home/user/lcx/ccpd_dataset/'
    label_dir = '../data/custom/labels'
    f = open('../data/custom/valid.txt')
    lines = f.readlines()
    path_length = len(path)
    for line in lines:
        tmp_path = line[path_length - 1: -4]
        label_path = label_dir + tmp_path + 'txt'
        new_label = genenrate_box(line[:-1]) # 最后有个/n
        # print(label_path)
        f2 = open(label_path, 'w')
        f2.write('0 ')
        for box in new_label:
            f2.write(str(box) + ' ')
        f2.close()
    f.close()

if __name__ == '__main__':
    # add_to_trainTxt()
    # generate_label_dir()
    generate_label()
