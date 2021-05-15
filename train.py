import os
import sys
from datetime import datetime

import paddle
from paddle.io import DataLoader
from paddle.static import InputSpec

from models.Loss import ClassLoss, BBoxLoss, LandmarkLoss, accuracy
from models.Net import RNet, ONet, PNet
from utils.data import CustomDataset

sys.path.append("../")
# 设置损失值的比例
radio_cls_loss = 1.0
radio_bbox_loss = 0.5

# 训练参数值
batch_size = 384
learning_rate = 1e-3
model_path = '../models_save'


# 开始训练
def train(model, radio_landmark_loss, epoch_num, data_path, name, shape):
    # 获取数据
    train_dataset = CustomDataset(data_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 设置优化方法
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[6, 14, 20], values=[0.001, 0.0001, 0.00001, 0.000001],
                                                   verbose=True)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=scheduler,
                                      weight_decay=paddle.regularizer.L2Decay(1e-4))

    # 获取损失函数
    class_loss = ClassLoss()
    bbox_loss = BBoxLoss()
    landmark_loss = LandmarkLoss()
    for epoch in range(epoch_num):
        for batch_id, (img, label, bbox, landmark) in enumerate(train_loader()):
            class_out, bbox_out, landmark_out = model(img)
            cls_loss = class_loss(class_out, label)
            box_loss = bbox_loss(bbox_out, bbox, label)
            landmarks_loss = landmark_loss(landmark_out, landmark, label)
            total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * box_loss + radio_landmark_loss * landmarks_loss
            total_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_id % 100 == 0:
                acc = accuracy(class_out, label)
                print('[%s] Train epoch %d, batch %d, total_loss: %f, cls_loss: %f, box_loss: %f, landmarks_loss: %f, '
                      'accuracy：%f' % (
                          datetime.now(), epoch, batch_id, total_loss, cls_loss, box_loss, landmarks_loss, acc))
        scheduler.step()

        # 保存模型
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        paddle.jit.save(layer=model,
                        path=os.path.join(model_path, name),
                        input_spec=[InputSpec(shape=shape, dtype='float32')])


# 获取O模型
modelO = ONet()
shapeO = [None, 3, 48, 48]
pathO = '../dataset/28/all_data'
paddle.summary(modelO, input_size=(batch_size, 3, 48, 48))
train(modelO, 0.5, 22, pathO, 'ONet', shapeO)

# 获取P模型
modelP = PNet()
shapeP = [None, 3, None, None]
pathP = '../dataset/12/all_data'
paddle.summary(modelP, input_size=(batch_size, 3, 12, 12))
train(modelP, 0.5, 30, pathP, 'PNet', shapeP)

# 获取R模型
modelR = RNet()
shapeR = [None, 3, 24, 24]
pathR = '../dataset/32/all_data'
paddle.summary(modelR, input_size=(batch_size, 3, 24, 24))
train(modelR, 0.5, 22, pathR, 'RNet', shapeR)
