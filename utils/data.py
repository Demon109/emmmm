import mmap
import os
import struct
import uuid

import cv2
import numpy as np
from paddle.io import Dataset
from tqdm import tqdm


class ImageData(object):
    def __init__(self, data_path):
        self.offset_dict = {}
        for line in open(data_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(data_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('正在加载数据标签...')
        # 获取label
        self.label = {}
        self.box = {}
        self.landmark = {}
        label_path = data_path + '.label'
        for line in open(label_path, 'rb'):
            key, bbox, landmark, label = line.split(b'\t')
            self.label[key] = int(label)
            self.box[key] = [float(x) for x in bbox.split()]
            self.landmark[key] = [float(x) for x in landmark.split()]
        print('数据加载完成，总数据量为：%d' % len(self.label))

    # 获取图像数据
    def get_img(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    # 获取图像标签
    def get_label(self, key):
        return self.label.get(key)

    # 获取人脸box
    def get_bbox(self, key):
        return self.box.get(key)

    # 获取关键点
    def get_landmark(self, key):
        return self.landmark.get(key)

    # 获取所有keys
    def get_keys(self):
        return self.label.keys()


def process(image):
    image = np.fromstring(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    assert (image is not None), 'image is None'
    # 把图片转换成numpy值
    image = np.array(image).astype(np.float32)
    # 转换成CHW
    image = image.transpose((2, 0, 1))
    # 归一化
    image = (image - 127.5) / 128
    return image


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_path):
        super(CustomDataset, self).__init__()
        self.imageData = ImageData(data_path)
        self.keys = self.imageData.get_keys()
        self.keys = list(self.keys)
        np.random.shuffle(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = self.imageData.get_img(key)
        assert (img is not None)
        label = self.imageData.get_label(key)
        assert (label is not None)
        bbox = self.imageData.get_bbox(key)
        landmark = self.imageData.get_landmark(key)
        img = process(img)
        label = np.array([label], np.int64)
        bbox = np.array(bbox, np.float32)
        landmark = np.array(landmark, np.float32)
        return img, label, bbox, landmark

    def __len__(self):
        return len(self.keys)


class DataSetWriter(object):
    def __init__(self, prefix):
        # 创建对应的数据文件
        self.data_file = open(prefix + '.data', 'wb')
        self.header_file = open(prefix + '.header', 'wb')
        self.label_file = open(prefix + '.label', 'wb')
        self.offset = 0
        self.header = ''

    def add_img(self, key, img):
        # 写入图像数据
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(img)))
        self.data_file.write(img)
        self.offset += 4 + len(key) + 4
        self.header = key + '\t' + str(self.offset) + '\t' + str(len(img)) + '\n'
        self.header_file.write(self.header.encode('ascii'))
        self.offset += len(img)

    def add_label(self, label):
        # 写入标签数据
        self.label_file.write(label.encode('ascii') + '\n'.encode('ascii'))


# 人脸识别训练数据的格式转换
def convert_data(data_folder, output_prefix):
    # 读取全部的数据类别获取数据
    data_list_path = os.path.join(data_folder, 'all_data_list.txt')
    train_list = open(data_list_path, "r").readlines()
    train_image_list = []
    for i, item in enumerate(train_list):
        sample = item.split(' ')
        # 获取图片路径
        image = sample[0]
        # 获取图片标签
        label = int(sample[1])
        # 做补0预操作
        bbox = [0, 0, 0, 0]
        landmark = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 如果只有box，关键点就补0
        if len(sample) == 6:
            bbox = [float(i) for i in sample[2:]]
        # 如果只有关键点，那么box就补0
        if len(sample) == 12:
            landmark = [float(i) for i in sample[2:]]
        # 加入到数据列表中
        train_image_list.append((image, label, bbox, landmark))
    print("训练数据大小：", len(train_image_list))

    # 开始写入数据
    writer = DataSetWriter(output_prefix)
    for image, label, bbox, landmark in tqdm(train_image_list):
        try:
            key = str(uuid.uuid1())
            img = cv2.imread(image)
            _, img = cv2.imencode('.bmp', img)
            # 写入对应的数据
            writer.add_img(key, img.tostring())
            label_str = str(label)
            bbox_str = ' '.join([str(x) for x in bbox])
            landmark_str = ' '.join([str(x) for x in landmark])
            writer.add_label('\t'.join([key, bbox_str, landmark_str, label_str]))
        except:
            continue
