import torch
import torch.nn.functional as F
import numpy as np
import os,time
import argparse
from scipy import misc
from imageio import imwrite,imsave
from Code.utils.dataloader_LungInf import test_dataset


def inference(pth_path,save_path):

    testsize = 352
    classes = 1
    data_path = './Dataset/TestingSet/LungInfection-Test/'
    pth_path = pth_path
    save_path = save_path

    from models import MMViT_seg_gated as net
    model = net.MMViTSeg(classes, aux=True)

    model.load_state_dict(torch.load(pth_path, map_location={'cuda:1':'cuda:0'}))
    model.cuda()
    model.eval()

    image_root = '{}/Imgs/'.format(data_path)
    # gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, testsize)
    os.makedirs(save_path, exist_ok=True)

    for i in range(test_loader.size):
        image, name = test_loader.load_data()
        image = image.cuda()
        # lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)
        pred1, pred2, pred3, pred4,_,_ = model(image)
        lateral_map_2 = pred1

        res = lateral_map_2
        # res = F.upsample(res, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # misc.imsave(opt.save_path + name, res)
        imsave(save_path + name, res)

    print('Test Done!')


if __name__ == "__main__":


    pth_base_path = './Snapshots/save_weights/MMViT-Seg/'
    save_base_path = './Results/Lung infection segmentation/'
    for i in range(40,100):
        print('i =', i)
        path =  pth_base_path + 'MMViT-Seg-' + str(i) + '.pth'
        print('path = ',path)
        pth_path = path
        save_path = save_base_path + 'MMViT-Seg-' + str(i) + '/'
        print('save_path = ', save_path)
        inference(pth_path,save_path)
        time.sleep(1)



