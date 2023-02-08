# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MMCNN(nn.Module):

    def __init__(self, args):
        super(MMCNN, self).__init__()
        self.sen_out_dim = args.sen_class
        self.emo_out_dim = args.emo_class

        # TextCNN
        self.text_conv2 = nn.Conv2d(1, 1, (2, 100))
        self.text_conv3 = nn.Conv2d(1, 1, (3, 100))
        self.text_conv4 = nn.Conv2d(1, 1, (4, 100))
        self.text_max2_pool = nn.MaxPool2d((args.sentence_max_size-2+1, 1))
        self.text_max3_pool = nn.MaxPool2d((args.sentence_max_size-3+1, 1))
        self.text_max4_pool = nn.MaxPool2d((args.sentence_max_size-4+1, 1))

        #AlexNet
        # input = 3 x 180 x 320
        self.alexnet = models.alexnet()
        self.img_linear = nn.Linear(1000, 3)
        
        self.sen_linear = nn.Linear(6, self.sen_out_dim)
        self.emo_linear = nn.Linear(6, self.emo_out_dim)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, img):
        # x.shape = batch_size x sentence_max_size x 100
        # img.shape = batch_size x 3 x 180 x 320
        x = x.unsqueeze(dim=1)
        batch = x.shape[0]
        # Convolution
        x1 = F.relu(self.text_conv2(x))
        x2 = F.relu(self.text_conv3(x))
        x3 = F.relu(self.text_conv4(x))

        # Pooling
        x1 = self.text_max2_pool(x1)
        x2 = self.text_max3_pool(x2)
        x3 = self.text_max4_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, -1)

        # dropout
        x = self.dropout(x)

        img = self.alexnet(img)
        img = self.dropout(img)

        img = self.img_linear(img)
        img = self.dropout(img)

        h = torch.cat([x, img], 1) # b x 6
        
        sen = self.sen_linear(h)
        sen = self.dropout(sen)
        
        emo = self.emo_linear(h)
        emo = self.dropout(emo)

        return sen, emo