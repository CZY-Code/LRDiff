import os
import random
import sys
import lmdb
import numpy as np
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():
    return transforms.ToPILImage()


def image_to_tensor():
    return transforms.ToTensor()


def image_to_edge(image, sigma):
    gray_image = rgb2gray(np.array(tensor_to_image()(image)))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))
    gray_image = image_to_tensor()(Image.fromarray(gray_image))
    return edge, gray_image


try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class GTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.GT_paths = None
        self.GT_env = None  # environment for lmdb
        self.GT_size = opt["GT_size"]
        self.origin_mask = opt["origin_mask"]
        self.origin_size = opt["origin_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.GT_paths, self.GT_sizes = util.get_image_paths(opt["data_type"], opt["dataroot_GT"])
        elif opt["data_type"] == "img":
            self.GT_paths = util.get_image_paths(opt["data_type"], opt["dataroot_GT"]) #GT list
            self.Mask_paths = util.get_image_paths(opt["data_type"], opt["dataroot_Mask"]) #mask list
            self.LQ_rootpath = opt["dataroot_LQ"]
            self.Mask_rootpath = opt["dataroot_Mask"]
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        print("dataset length: {}".format(len(self.GT_paths)))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if self.GT_env is None:
                self._init_lmdb()

        # get GT image
        GT_path = self.GT_paths[index]
        img_name = os.path.basename(GT_path)
        
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
            
        img_GT = util.read_img(self.GT_env, GT_path, resolution) #return: Numpy float32, HWC, BGR, [0,1]
        img_LQ = util.read_img(self.GT_env, os.path.join(self.LQ_rootpath, img_name), resolution)

        if self.origin_mask:
            img_Mask = cv2.imread(os.path.join(self.Mask_rootpath, img_name), cv2.IMREAD_GRAYSCALE)
        else: #不同iter使用不同掩码会使得的mu不同，导致对同一个x_T的去噪训练不连续，因此而无法收敛？还是数据扩张平方倍需要的训练时间加长
            img_Mask = cv2.imread(random.choice(self.Mask_paths), cv2.IMREAD_GRAYSCALE)

        img_GT = cv2.resize(img_GT, (self.origin_size, self.origin_size), interpolation=cv2.INTER_NEAREST)
        img_LQ = cv2.resize(img_LQ, (self.origin_size, self.origin_size), interpolation=cv2.INTER_NEAREST)
        img_Mask = cv2.resize(img_Mask, (self.origin_size, self.origin_size), interpolation=cv2.INTER_NEAREST) #[768, 768]
        
        if self.opt["segment"]: #训练时用于减少内存占用
            rnd_h = random.randint(0, max(0, self.origin_size - self.GT_size))
            rnd_w = random.randint(0, max(0, self.origin_size - self.GT_size))
            img_GT = img_GT[rnd_h : rnd_h + self.GT_size, rnd_w : rnd_w + self.GT_size, :]
            img_LQ = img_LQ[rnd_h : rnd_h + self.GT_size, rnd_w : rnd_w + self.GT_size, :]
            img_Mask = img_Mask[rnd_h : rnd_h + self.GT_size, rnd_w : rnd_w + self.GT_size] #截取mask的一部分，增加这行会使得总缺损变少

        img_Mask = 1 - img_Mask[..., np.newaxis] / 255.0 #[256, 256, 1], 0 is masked, 1 is unmasked, 需要阈值二值化吗？
        if self.opt["phase"] == "train": # augmentation - flip, rotate
            img_GT, img_LQ, img_Mask = util.augment4imgs([img_GT, img_LQ, img_Mask], self.opt["use_flip"], self.opt["use_rot"], self.opt["mode"])
        img_Mask = np.where(img_Mask>=0.5, 1.0, 0.0) #二值化
        img_LQ = img_LQ * img_Mask

        # change color space if necessary
        if self.opt["color"]:
            img_GT, img_LQ = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT, img_LQ])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[..., ::-1]
            img_LQ = img_LQ[..., ::-1]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float() #HWC->CHW
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_Mask = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Mask, (2, 0, 1)))).float() #HWC->CHW
        GT_edge, GT_gray = image_to_edge(img_GT, sigma=3.)

        return {"GT": img_GT, "LQ": img_LQ, "Mask": img_Mask, "GT_path": GT_path, "GT_edge": GT_edge, "GT_gray": GT_gray}

    def __len__(self):
        return len(self.GT_paths)
