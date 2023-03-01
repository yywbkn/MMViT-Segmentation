
import torch
from torch.autograd import Variable
import os
import random
import numpy as np
import argparse
from datetime import datetime
from Code.utils.dataloader_LungInf import get_loader
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F


def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return ( 0.2 * wbce + 0.8* wiou).mean()


def train(train_loader, model, optimizer, epoch, train_save):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 0.8, 1, 1.2,1.25]    # replace your desired scale, try larger scale for better accuracy in small object
    loss_record0, loss_record1, loss_record2, loss_record3, loss_record4, loss_record5, loss_record6 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = Variable(images).cuda()

            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()

            # ---- rescaling the inputs (img/gt/edge) ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.upsample(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_4, edge01, edge02 = model(images)

            # ---- loss function ----
            loss5 = joint_loss(lateral_map_4, gts)
            loss4 = joint_loss(lateral_map_3, gts)
            loss3 = joint_loss(lateral_map_2, gts)
            loss2 = joint_loss(lateral_map_1, gts)
            loss1 = torch.nn.BCEWithLogitsLoss()(edge02, edges)
            loss0 = torch.nn.BCEWithLogitsLoss()(edge01, edges)

            loss = 12 * loss0 + 12 * loss1 + 5 * loss2 + 2 * loss3 + 2 *loss4 + loss5
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record0.update(loss0.data, opt.batchsize)
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
                loss_record6.update(loss2.data, opt.batchsize)
        # ---- train logging ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-edge: {:.4f}, lateral-1: {:.4f}, lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f},lateral-6: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record0.show(),loss_record1.show(),
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show(),loss_record6.show()))
    # ---- save model_lung_infection ----
    save_path = './Snapshots/save_weights/{}/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), save_path + 'MMViT-Seg-%d.pth' % (epoch+1))
        print('[Saving Snapshot:]', save_path + 'MMViT-Seg-%d.pth' % (epoch+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--epoch', type=int, default=200,help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
    parser.add_argument('--classes', type=int, default=1, help='No. of classes in the dataset')
    parser.add_argument('--batchsize', type=int, default=2,help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,help='set the size of training sample')
    parser.add_argument('--clip', type=float, default=0.5,help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,help='every n epochs decay learning rate')
    parser.add_argument('--gpu_device', type=int, default=0,help='choose which GPU device you want to use')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers in dataloader. In windows, set num_workers=0')
    # model_lung_infection parameters
    parser.add_argument('--net_channel', type=int, default=32,help='internal channel numbers in the MMViT-Seg, default=32, try larger for better accuracy')
    parser.add_argument('--n_classes', type=int, default=1,help='binary segmentation when n_classes=1')
    # training dataset
    parser.add_argument('--train_path', type=str,default='./Dataset/TrainingSet/LungInfection-Train/Doctor-label')
    parser.add_argument('--train_save', type=str, default='MMViT-Seg')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # ---- build models ----
    torch.cuda.set_device(opt.gpu_device)

    from models import MMViT_seg_gated as net

    model = net.MMViTSeg(opt.classes, aux=True)
    model = model.cuda()


    # ---- load training sub-modules ----
    BCE = torch.nn.BCEWithLogitsLoss()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root,batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.num_workers)
    total_step = len(train_loader)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.train_save)



