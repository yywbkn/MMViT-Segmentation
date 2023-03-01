

import torch
from torch import nn
# from module_helper import ModuleHelper
from .module_helper import ModuleHelper
from torch.nn import functional as F


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(2, 3), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class Multi_query_attention(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, in_channels02, in_channels03, in_channels04,out_channels=None,norm_type=None,):
        super(Multi_query_attention, self).__init__()
        self.in_channels = in_channels
        self.in_channels02 = in_channels02
        self.in_channels03 = in_channels03
        self.in_channels04 = in_channels04
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
        )

        self.f_query = self.f_key

        self.f_value = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
        )

        self.f_query02 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels02, out_channels=self.key_channels, kernel_size=1, stride=1,padding=0),
            ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
        )

        self.f_query03 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels03, out_channels=self.key_channels, kernel_size=1,stride=1,padding=0),
            ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
        )

        self.f_query04 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels04, out_channels=self.key_channels, kernel_size=1, stride=1,padding=0),
            ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
        )

        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        psp_size01 = (1, 3)
        psp_size02 = (1, 2)
        psp_size03 = (2, 4)
        self.psp01 = PSPModule(psp_size01)
        self.psp02 = PSPModule(psp_size02)
        self.psp03 = PSPModule(psp_size03)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x1, x2, x3, x4):
        batch_size, h, w = x1.size(0), x1.size(2), x1.size(3)
        value1x1 = self.f_value(x1)
        value = self.psp01(value1x1)

        query1x1 = self.f_query(x1)
        Q1 = self.psp01(query1x1)

        key1x1 = self.f_key(x1)
        K1 = self.psp01(key1x1)

        # first affinity_map [SX C] X [C X S] = [S X S]
        Q1 = Q1.permute(0, 2, 1)
        affinity_map = torch.matmul(Q1, K1)

        # first  normlize
        affinity_map = (self.key_channels ** -.5) * affinity_map
        affinity_map = F.softmax(affinity_map, dim=-1)

        # first QKV1 [S X S] X [S X C] = [S X C]
        V = value.permute(0, 2, 1)  # [S X C]
        QKV1 = torch.matmul(affinity_map, V)

        # second  affinity_map [S2 X C] X [C X S] = [S2 X S]
        query1x1_02 = self.f_query02(x2)
        Q2 = self.psp02(query1x1_02)

        Q2 = Q2.permute(0, 2, 1)
        QKV1 = QKV1.permute(0, 2, 1)
        QK2 = torch.matmul(Q2, QKV1)

        # second normlize
        QK2 = (self.key_channels ** -.5) * QK2
        QK2 = F.softmax(QK2, dim=-1)

        # second QKV1 [S2 X S] X [S X C] = [S2 X C]
        QKV1 = QKV1.permute(0, 2, 1)
        QKV2 = torch.matmul(QK2, QKV1)


        #  third affinity_map [N X C] X [C X S3] = [N x S3]
        query1x1_03 = self.f_query03(x3)
        Q3 = self.psp03(query1x1_03)

        Q3 = Q3.permute(0, 2, 1)
        QKV2 = QKV2.permute(0, 2, 1)
        QK3 = torch.matmul(Q3, QKV2)

        # third normlize
        QK3 = (self.key_channels ** -.5) * QK3
        QK3 = F.softmax(QK3, dim=-1)

        # third QKV3 [N X S3] X [S3 X C] = [N X C]
        QKV2 = QKV2.permute(0, 2, 1)
        QKV3 = torch.matmul(QK3, QKV2)


        #   affinity_map [N X C] X [C X Sn] = [N x Sn]
        query1x1_04 = self.f_query04(x4)
        Qn = query1x1_04
        Qn = Qn.view(batch_size, self.key_channels, -1)
        Qn = Qn.permute(0, 2, 1)
        QKV3 = QKV3.permute(0, 2, 1)
        QKn = torch.matmul(Qn, QKV3)

        # n normlize
        QKn = (self.key_channels ** -.5) * QKn
        QKn = F.softmax(QKn, dim=-1)

        # n QKVn [N X S2] X [S2 X C] = [N X C]
        QKV3 = QKV3.permute(0, 2, 1)
        QKVn = torch.matmul(QKn, QKV3)

        context = QKVn.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x4.size()[2:])
        context = self.W(context)

        return context

