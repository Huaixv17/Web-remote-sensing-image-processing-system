#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 划分训练集/验证集/测试集，并生成文件名列表
# 所有样本从RSOD数据集的playground子集中选取

import random
import os.path as osp
from os import listdir


# 随机数生成器种子
RNG_SEED = 52980
# 调节此参数控制训练集数据的占比
TRAIN_RATIO = 0.9
# 调节此参数控制验证集数据的占比
VAL_RATIO = 0.05
# 数据集路径
DATA_DIR = 'C:/Users/Administrator/diwu_pre/baidu/jice/data/'

# 目标类别
CLASS = 'playground'

print("数据集划分已完成。")


# In[2]:


# 导入需要用到的库

import random
import os.path as osp

import cv2
import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
from paddlers.tasks.utils.visualize import visualize_detection
from matplotlib import pyplot as plt
from PIL import Image


# In[3]:


# 定义全局变量

# 随机种子
SEED = 52980
# 数据集存放目录
DATA_DIR = 'C:/Users/Administrator/diwu_pre/baidu/jice/data/'
# 训练集`file_list`文件路径

# 验证集`file_list`文件路径

# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = 'C:/Users/Administrator/diwu_pre/baidu/jice/data/test.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = 'C:/Users/Administrator/diwu_pre/baidu/jice/data/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR =  'C:/Users/Administrator/diwu_pre/baidu/jice/exp/'
# 目标类别
CLASS = 'playground'
# 模型验证阶段输入影像尺寸
INPUT_SIZE = 608


# In[4]:


# 固定随机种子，尽可能使实验结果可复现

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)


# In[5]:


# 构建数据集


eval_transforms = T.Compose([
    # 使用双三次插值将输入影像缩放到固定大小
    T.Resize(
        target_size=INPUT_SIZE, interp='CUBIC'
    ),
    # 验证阶段与训练阶段的归一化方式必须相同
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])



# 分别构建训练和验证所用的数据集


# In[6]:


# 构建PP-YOLO模型
model = pdrs.tasks.PPYOLO(num_classes=1)
model.net_initialize(
    pretrain_weights='COCO',
    save_dir=osp.join(EXP_DIR, 'pretrain'),
    resume_checkpoint=None,
    is_backbone_weights=False
)


# In[8]:


# 构建测试集
test_dataset = pdrs.datasets.VOCDetection(
    data_dir=DATA_DIR,
    file_list=TEST_FILE_LIST_PATH,
    label_list=LABEL_LIST_PATH,
    transforms=eval_transforms,
    shuffle=False
)


# 为模型加载历史最佳权重
state_dict = paddle.load(osp.join(EXP_DIR, 'model.pdparams'))
model.net.set_state_dict(state_dict)
print(test_dataset)
# 执行测试
test_result = model.evaluate(test_dataset)
print(
    "测试集上指标：bbox mAP为{:.2f}".format(
        test_result['bbox_map'],
    )
)


# In[12]:


# 预测结果可视化
# 重复运行本单元可以查看不同结果

def read_rgb(path):
    im = cv2.imread(path)
    im = im[...,::-1]
    return im


def show_images_in_row(ims, fig, title='', lut=None):
    n = len(ims)
    fig.suptitle(title)
    axs = fig.subplots(nrows=1, ncols=n)
    for idx, (im, ax) in enumerate(zip(ims, axs)):
        # 去掉刻度线和边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        ax.imshow(im)

def get_lut():
    lut = np.zeros((256,3), dtype=np.uint8)
    lut[0] = [255, 0, 0]
    lut[1] = [30, 255, 142]
    lut[2] = [60, 0, 255]
    lut[3] = [255, 222, 0]
    lut[4] = [0, 0, 0]
    return lut
        
# 需要展示的样本个数
num_imgs_to_show = 4
# 随机抽取样本
chosen_indices = random.choices(range(len(test_dataset)), k=num_imgs_to_show)

# 参考 https://stackoverflow.com/a/68209152
fig = plt.figure(constrained_layout=True)
fig.suptitle("Test Results")

subfigs = fig.subfigures(nrows=2, ncols=1)

# 读取输入影像并显示
ims = [read_rgb(test_dataset.file_list[idx]['image']) for idx in chosen_indices]
show_images_in_row(ims, subfigs[0], title='Image')

# 绘制目标框
with paddle.no_grad():
    vis_res = []
    model.labels = test_dataset.labels
    for idx, im in zip(chosen_indices, ims):
        sample = test_dataset[idx]
        gt = [
            {
                'category_id': cid[0], 
                'category': CLASS, 
                'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], 
                'score': 1.0
            } 
            for cid, bbox in zip(sample['gt_class'], sample['gt_bbox'])
        ]

        im = cv2.resize(im[...,::-1], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        pred = model.predict(im, eval_transforms)

        vis = im
        # 用绿色画出预测目标框
        if len(pred) > 0:
            vis = visualize_detection(
                np.array(vis), pred, 
                color=np.asarray([[0,255,0]], dtype=np.uint8), 
                threshold=0.2, save_dir=None
            )
        # 用蓝色画出真实目标框
        if len(gt) > 0:
            vis = visualize_detection(
                np.array(vis), gt, 
                color=np.asarray([[0,0,255]], dtype=np.uint8), 
                save_dir=None
            )
        vis_res.append(vis)
show_images_in_row(vis_res, subfigs[1], title='Detection')
im = Image.fromarray(get_lut()[vis_res[0]])
im.save("jice/results/your_file.png")
# 渲染结果
fig.canvas.draw()
Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())


# In[ ]:




