# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The SimDR and SA-SimDR part:
# Written by Yanjie Li (lyj20@mails.tsinghua.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

import ipdb

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, coord_representation='heatmap', simdr_split_ratio=1):
		##人体关键点的数目
        self.num_joints = 0
        ##像素标准化参数
        self.pixel_std = 200
        ##水平翻转
        self.flip_pairs = []
        ##父母ID??
        self.parent_ids = []

        ##是否进行训练
        self.is_train = is_train
        ##训练数据根目录
        self.root = root
        ##图片数据集名称，如'train2017'
        self.image_set = image_set

        ##输出目录
        self.output_path = cfg.OUTPUT_DIR
        ##数据格式如'jpg'
        self.data_format = cfg.DATASET.DATA_FORMAT

        ##缩放因子
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        ##旋转角度
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        ##是否进行水平翻转
        self.flip = cfg.DATASET.FLIP
        ##人体一半的关键点数目，默认为8
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        ##人体一半的概率
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        ##是否转换成RGB的图片格式，默认为true；注意一定要转成RGB
        self.color_rgb = cfg.DATASET.COLOR_RGB

		##神经网络输入的训练图片的大小，如[192, 256]；大多数情况下人是站立的
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        ##heatmap的大小，即神经网络输出的特征图的大小，如[48, 64]
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        ##sigma参数，默认为2
        self.sigma = cfg.MODEL.SIGMA
        ##是否对每个关键点使用不同的权重
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        ##关键点权重
        self.joints_weight = 1

        ##数据增强、转换等
        self.transform = transform
        ##用于保存训练数据的信息，由子类提供，例如class COCODataset(JointsDataset)
        self.db = []

        # simdr related
        self.coord_representation = coord_representation
        ##用于提升关键点精度的分离因子，对应文中的公式(2)
        self.simdr_split_ratio = simdr_split_ratio
        self.loss = cfg.LOSS.TYPE
        assert self.coord_representation in ['simdr', 'sa-simdr', 'heatmap'], 'only simdr or sa-simdr or heamtap supported'

	##由子类实现
    def _get_db(self):
        raise NotImplementedError

    ##由子类实现
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        """
        只有一半身体的数据转换
        :param joints: 关键点位置，shape=[17,3]，因为使用2D表示，第三维度都为0
        :param joints_vis: 表示关键点是否可见，shape=[17,3]
        :return:
        """
        ##上半部分的关键点
        upper_joints = []
        ##下半部分的关键点
        lower_joints = []
        for joint_id in range(self.num_joints):
            ##如果该关键点可见
            if joints_vis[joint_id][0] > 0:
                ##如果关键点为上半身的关键点
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                ##如果关键点为下半身的关键点
                else:
                    lower_joints.append(joints[joint_id])

        ##二分之一的概率进行关键点选择，选择上半身或者下半身关键点
        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        ##如果该样本的关键点小于两个，则返回None，无需进行训练
        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)

        ##求得关键点x、y的平均坐标
        center = selected_joints.mean(axis=0)[:2]

        ##左上角坐标
        left_top = np.amin(selected_joints, axis=0)
        ##右下角坐标
        right_bottom = np.amax(selected_joints, axis=0)

        ##获得包揽所有关键点的最小宽和高
        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        ##对w或者h进行扩大，保持纵横比
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        ##图像w、h的缩放比例
        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    ##所有Dataset抽象类的子类应该override__len__和__getitem__
    ##__len__提供数据集的大小
    def __len__(self,):
        return len(self.db)

    ##所有Dataset抽象类的子类应该override__len__和__getitem__
    ##__getitem__支持整数索引，范围从0到len(self)
    def __getitem__(self, idx):
        ##根据idx从db获取样本信息
        db_rec = copy.deepcopy(self.db[idx])
        ##获取图像名
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        # print(image_file)

		##如果数据格式为zip则解压
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        ##否则直接读取图像，获得像素值
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        ##是否转化为rgb格式，注意一定要转换成rgb
        if self.color_rgb:
            # print(data_numpy)
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        ##如果读取到的数据不为numpy格式则报错
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        ##获取人体关键点坐标
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        ##获取训练图像转换所需的center以及scale
        c = db_rec['center']
        s = db_rec['scale']

        ##如果训练样本中没有设置score，则加载该属性，并且设置为1
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        ##如果是进行训练
        if self.is_train:
            ##如果可见关键点大于人体一半关键点，并且生成的随机数小于self.prob_half_body=0.3
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                ##重新调整center、scale
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            ##缩放因子scale_factor=0.35，以及旋转角度rotation_factor=45
            sf = self.scale_factor
            rf = self.rotation_factor

            ##s大小为[1-0.35=0.65, 1+0.35=1.35]之间
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            ##r大小为[-2*45=90, 2*45=90]之间
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            ##进行数据水平翻转
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        ##进行正则化、形状改变等
        if self.transform:
            input = self.transform(input)

        ##对人体关键点的坐标也进行仿射变换
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        ##获得ground truth，target热图的维度为[17, 64, 48]，target_weight的维度为[17, 1]
        target, target_weight = self.generate_target(joints, joints_vis)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        # ipdb.set_trace()

        if self.coord_representation == 'simdr':
            ##joints_split为numpy，shape为(17, 3)
            joints_split = joints.copy()
            joints_split = np.around(joints_split * self.simdr_split_ratio)
            joints_split = joints_split.astype(np.int64)
            target_weight,filtered_joints = self.filter_target_simdr(joints_split.copy(), joints_vis, self.image_size*self.simdr_split_ratio)
            # ipdb.set_trace()
            ##filtered_joints.shape为(17, 3)；filtered_joints[:,0:2].shape为(17, 2)
            ##target_weight.shape为(17, 1)
            return input, filtered_joints[:,0:2], target_weight, meta
        elif self.coord_representation == 'sa-simdr':
            target_x, target_y, target_weight = self.generate_sa_simdr(joints, joints_vis)
            target_x = torch.from_numpy(target_x)
            target_y = torch.from_numpy(target_y)
            target_weight = torch.from_numpy(target_weight)
            ##target_x.shape为(17, 384), cfg.MODEL.IMAGE_SIZE[0] * cfg.MODEL.SIMDR_SPLIT_RATIO = 192 * 2
            ##target_y.shape为(17, 512), cfg.MODEL.IMAGE_SIZE[1] * cfg.MODEL.SIMDR_SPLIT_RATIO = 256 * 2
            ##target_weight.shape为(17, 1)
            return input, target_x, target_y, target_weight, meta            
        elif self.coord_representation == 'heatmap':
            return input, target, target_weight, meta

    def generate_sa_simdr(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target_x = np.zeros((self.num_joints,
                            int(self.image_size[0]*self.simdr_split_ratio)),
                            dtype=np.float32)
        target_y = np.zeros((self.num_joints,
                            int(self.image_size[1]*self.simdr_split_ratio)),
                            dtype=np.float32)                              

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            target_weight[joint_id] = \
                self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
            if target_weight[joint_id] == 0:
                continue

            mu_x = joints[joint_id][0] * self.simdr_split_ratio
            mu_y = joints[joint_id][1] * self.simdr_split_ratio
            
            x = np.arange(0, int(self.image_size[0]*self.simdr_split_ratio), 1, np.float32)
            y = np.arange(0, int(self.image_size[1]*self.simdr_split_ratio), 1, np.float32)

            v = target_weight[joint_id]
            if v > 0.5:
                target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * self.sigma ** 2)))/(self.sigma*np.sqrt(np.pi*2))
                target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * self.sigma ** 2)))/(self.sigma*np.sqrt(np.pi*2))
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        # ipdb.set_trace()
        ##target_x.shape为(17, 384), cfg.MODEL.IMAGE_SIZE[0] * cfg.MODEL.SIMDR_SPLIT_RATIO = 192 * 2
        ##target_y.shape为(17, 512), cfg.MODEL.IMAGE_SIZE[1] * cfg.MODEL.SIMDR_SPLIT_RATIO = 256 * 2
        ##target_weight.shape为(17, 1)
        return target_x, target_y, target_weight  

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected   

	##生成关键点热图
    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        ##target_weight的维度为[17, 1]
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((self.num_joints,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)

		##self.sigma 默认为2，tmp_size=6
        tmp_size = self.sigma * 3

		##为每个关键点生成target热图以及对应的热图权重target_weight
        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ##根据tmp_size参数，计算出关键点范围左上角和右下角坐标
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            
            ##判断该关键点是否处于热图之外；如果处于热图之外，则把该热图对应的target_weight设置为0，然后continue
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32) # [size]
            y = x[:, np.newaxis] # [size,1]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)) # [size] 

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def filter_target_simdr(self, joints, joints_vis, image_size):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :param image_size: image_size
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0].copy()

        # detect abnormal coords and make the weight 0
        for joint_id in range(self.num_joints):
            if joints[joint_id][1] < 0:
                target_weight[joint_id] = 0
                joints[joint_id][1]=0
            elif joints[joint_id][1] >= image_size[1]:
                target_weight[joint_id] = 0
                joints[joint_id][1] = image_size[1] - 1
            if joints[joint_id][0] < 0:
                target_weight[joint_id] = 0
                joints[joint_id][0] = 0
            elif joints[joint_id][0] >= image_size[0]:
                target_weight[joint_id] = 0
                joints[joint_id][0] = image_size[0] - 1

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target_weight,joints  

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= (self.image_size[0]*self.simdr_split_ratio) or ul[1] >= self.image_size[1]*self.simdr_split_ratio \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight
        

