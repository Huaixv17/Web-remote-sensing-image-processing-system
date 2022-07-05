#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 划分训练集/验证集/测试集，并生成文件名列表

import random
import os.path as osp
from os import listdir

import cv2


# 随机数生成器种子
RNG_SEED = 56961
# 调节此参数控制训练集数据的占比
TRAIN_RATIO = 0.9
# 数据集路径
DATA_DIR = 'C:/Users/Administrator/diwu_pre/baidu/tiqu/data/'

# 分割类别
CLASSES = (
    'background',
    'road',
)

print("数据集划分已完成。")


# In[4]:


# 将GT中的255改写为1，便于训练

import os.path as osp
from glob import glob

import cv2
from tqdm import tqdm


# In[5]:


# 导入需要用到的库

import random
import os.path as osp

import cv2
import numpy as np
import paddle
import paddlers as pdrs
from paddlers import transforms as T
from matplotlib import pyplot as plt
from PIL import Image


# In[6]:


# 定义全局变量

# 随机种子
SEED = 56961
# 数据集存放目录
DATA_DIR = 'C:/Users/Administrator/diwu_pre/baidu/tiqu/data'
# 训练集`file_list`文件路径

# 验证集`file_list`文件路径

# 测试集`file_list`文件路径
TEST_FILE_LIST_PATH = 'C:/Users/Administrator/diwu_pre/baidu/tiqu/data/test.txt'
# 数据集类别信息文件路径

# 实验目录，保存输出的模型权重和结果
EXP_DIR =  'C:/Users/Administrator/diwu_pre/baidu/tiqu/exp/'


# In[7]:


# 固定随机种子，尽可能使实验结果可复现

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)


# In[8]:


# 构建数据集

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
eval_transforms = T.Compose([
    T.Resize(target_size=1488),
    # 验证阶段与训练阶段的数据归一化方式必须相同
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 分别构建训练和验证所用的数据集


# In[9]:


# 构建DeepLab V3+模型，使用ResNet-50作为backbone
model = pdrs.tasks.DeepLabV3P(
    input_channel=3,
    
    backbone='ResNet50_vd'
)
model.net_initialize(
    pretrain_weights='CITYSCAPES',
    save_dir=osp.join(EXP_DIR, 'pretrain'),
    resume_checkpoint=None,
    is_backbone_weights=False
)


# In[10]:


# 构建优化器
optimizer = paddle.optimizer.Adam(
    learning_rate=0.001, 
    parameters=model.net.parameters()
)


# In[11]:


# 构建测试集
test_dataset = pdrs.datasets.SegDataset(
    data_dir=DATA_DIR,
    file_list=TEST_FILE_LIST_PATH,

    transforms=eval_transforms,
    num_workers=0,
    shuffle=False
)


# 为模型加载历史最佳权重
state_dict = paddle.load(osp.join(EXP_DIR, 'model.pdparams'))
model.net.set_state_dict(state_dict)

# 执行测试
test_result = model.evaluate(test_dataset)
print(
    "测试集上指标：IoU为{:.2f}，Acc为{:.2f}，Kappa系数为{:.2f}, F1为{:.2f}".format(
        test_result['category_iou'][1], 
        test_result['category_acc'][1],
        test_result['kappa'],
        test_result['category_F1-score'][1]
    )
)


# In[19]:


# 预测结果可视化
# 重复运行本单元可以查看不同结果

def read_image(path):
    im = cv2.imread(path)
    return im[...,::-1]


def show_images_in_row(ims, fig, title='', quantize=False):
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

        if isinstance(im, str):
            im = read_image(im)

        if quantize:
            im = (im*255).astype('uint8')

        if im.ndim == 2:
            im = np.tile(im[...,np.newaxis], [1,1,3])

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
num_imgs_to_show = 2
# 随机抽取样本
chosen_indices = random.choices(range(len(test_dataset)), k=num_imgs_to_show)

# 参考 https://stackoverflow.com/a/68209152
fig = plt.figure(constrained_layout=True)
fig.suptitle("Test Results")

subfigs = fig.subfigures(nrows=3, ncols=1)

# 读取输入影像并显示
im_paths = [test_dataset.file_list[idx]['image'] for idx in chosen_indices]
show_images_in_row(im_paths, subfigs[0], title='Image')

# 获取模型预测输出
with paddle.no_grad():
    model.net.eval()
    preds = []
    for idx in chosen_indices:
        input, mask = test_dataset[idx]
        input = paddle.to_tensor(input).unsqueeze(0)
        logits, *_ = model.net(input)
        pred = paddle.argmax(logits[0], axis=0)
        preds.append(pred.numpy())
show_images_in_row(preds, subfigs[1], title='Pred', quantize=True)

im = Image.fromarray(get_lut()[preds[0]])
im.save("tiqu/results/your_file.png")
# 读取真值标签并显示


# 渲染结果
fig.canvas.draw()
Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())


# In[ ]:




