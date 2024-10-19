import os
import numpy as np
import cv2
from tqdm import tqdm
import tdm.utils as util


def compare_images(folder_pred, folder_GT):
    """比较文件夹 folder_pred 和 folder_GT 中相同命名的图像"""
    test_PSNR = []
    test_SSIM = []
    for filename in tqdm(os.listdir(folder_pred)):
        path_pred = os.path.join(folder_pred, filename)
        if filename.endswith('.png'):
            base_name = filename[:-4]  # 去掉 _f.png 后缀
            path_GT = os.path.join(folder_GT, base_name + '.jpg')  # 生成对应的 .jpg 路径
        else:
            continue
        # 检查文件是否在两个文件夹中都存在
        if os.path.isfile(path_GT):
            # 读取图像
            img_a = cv2.resize(cv2.imread(path_pred, cv2.IMREAD_UNCHANGED), (768, 768), interpolation=cv2.INTER_NEAREST)
            img_b = cv2.resize(cv2.imread(path_GT, cv2.IMREAD_UNCHANGED), (768, 768), interpolation=cv2.INTER_NEAREST)

            # 确保图像读取成功
            if img_a is not None and img_b is not None:
                # 计算 PSNR
                test_PSNR.append(util.calculate_psnr(img_a, img_b))
                test_SSIM.append(util.calculate_ssim(img_a, img_b))
            else:
                print(f'Error reading images: {filename}')
        else:
            print(f'File not found in folder B: {filename}')
    print(f"PSNR: {np.mean(test_PSNR)}, SSIM: {np.mean(test_SSIM)}")

# 定义文件夹路径
folder_pred = './dataset/sd-31/'
# folder_pred = '/home/chengzy/mural-completion/tdm/results/ir-sde/Val_Dataset/'
folder_GT = './dataset/muralV3test/Images/'
# 进行比较
compare_images(folder_pred, folder_GT)
