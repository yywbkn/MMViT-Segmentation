
from PIL import Image
import os
import numpy as np


def split_class(pred_path, img_name, w_gt, h_gt):
    im = Image.open(os.path.join(pred_path, img_name)).convert('L')
    im_array_red = np.array(im)  # 0, 38
    im_array_green = np.array(im)  # 0, 75
    print(np.unique(im_array_red))

    im_array_red[im_array_red != 0] = 1
    im_array_red[im_array_red == 0] = 255
    im_array_red[im_array_red == 1] = 0

    im_array_green[im_array_green != 127] = 0
    im_array_green[im_array_green == 127] = 255

    cls_save_1 = pred_path.replace('class_12', 'Ground-glass opacities/MulCls_MMViTSeg')
    cls_save_2 = pred_path.replace('class_12', 'Consolidation/MulCls_MMViTSeg')
    os.makedirs(cls_save_1, exist_ok=True)
    os.makedirs(cls_save_2, exist_ok=True)

    Image.fromarray(im_array_red).convert('1').resize(size=(h_gt, w_gt)).save(os.path.join(cls_save_1, img_name))
    Image.fromarray(im_array_green).convert('1').resize(size=(h_gt, w_gt)).save(os.path.join(cls_save_2, img_name))