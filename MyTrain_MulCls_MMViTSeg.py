
import argparse
import os
import numpy as np
import torch.optim as optim
from Code.utils.dataloader_MulClsLungInf_UNet import LungDataset
from torchvision import transforms
# from LungData import test_dataloader, train_dataloader  # pls change batch_size
from torch.utils.data import DataLoader
import  torch
from models import MMViT_seg_gated_MulCls as net
import random
from torch import nn

def train(epo_num, input_channels, batch_size, lr, save_path):
    train_dataset = LungDataset(
        imgs_path='./Dataset/TrainingSet/MultiClassInfection-Train/Imgs/',
        # NOTES: prior is borrowed from the object-level label of train split
        pseudo_path='./Dataset/TrainingSet/MultiClassInfection-Train/Prior/',
        label_path='./Dataset/TrainingSet/MultiClassInfection-Train/GT/',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    lung_model = net.MMViTSeg(classes = 3, aux=False)
    print(lung_model)
    lung_model = lung_model.to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(lung_model.parameters(), lr=lr, momentum=0.7)

    for epo in range(epo_num):

        train_loss = 0
        lung_model.train()

        for index, (img, pseudo, img_mask, _) in enumerate(train_dataloader):

            img = img.to(device)
            pseudo = pseudo.to(device)
            img_mask = img_mask.to(device)

            optimizer.zero_grad()
            output = lung_model(torch.cat((img, pseudo), dim=1))

            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, img_mask)

            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()

            if np.mod(index, 20) == 0:
                print('Epoch: {}/{}, Step: {}/{}, Train loss is {}'.format(epo, epo_num, index, len(train_dataloader), iter_loss))

        os.makedirs('./checkpoints//MulCls_MMViTSeg', exist_ok=True)
        if np.mod(epo+1, 10) == 0:
            torch.save(lung_model.state_dict(),'./Snapshots/save_weights/{}/MulCls_MMViTSeg_{}.pkl'.format(save_path, epo+1))
            print('Saving checkpoints: MulCls_MMViTSeg_{}.pkl'.format(epo+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    train(epo_num=500,
          input_channels=6,
          batch_size=2,
          lr=0.8e-3,
          save_path='MulCls_MMViTSeg')
