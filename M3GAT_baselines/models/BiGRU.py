# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BiGRU(nn.Module):

    def __init__(self, args):
        super(BiGRU, self).__init__()
        self.m = 8
        self.sentence_max_size = args.sentence_max_size
        self.sen_out_dim = args.sen_class
        self.emo_out_dim = args.emo_class

        self.gru = nn.GRU(input_size=100,
                          hidden_size=100,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)

        self.text_cnn = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(self.m, 100))
        self.linear3 = nn.Linear(100, 100)

        self.lstm = nn.LSTM(input_size=100,
                            hidden_size=100,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        #AlexNet
        # input = 3 x 180 x 320
        self.alexnet = models.alexnet()
        self.img_linear = nn.Linear(1000, 100)
        
        self.sen_linear = nn.Linear(200, self.sen_out_dim)
        self.emo_linear = nn.Linear(200, self.emo_out_dim)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, img):
        # x.shape = batch_size x sentence_max_size x 100
        # img.shape = batch_size x 3 x 180 x 320
        d = x.shape[-1]
        h = []

        for i in range(self.sentence_max_size-self.m+1):
            local = x[:, i:i+self.m, :] # local.shape = batch_size x m x 100
            local_gru, _ = self.gru(local) # local_gru.shape = batch_size x m x 200

            local_f = local_gru[:,-1,:].squeeze() # local_f.shape = batch_size x 200
            local_f = local_f[:, :d] # local_f.shape = batch_size x 100

            local_b = local_gru[:,0,:].squeeze() # local_b.shape = batch_size x 200
            local_b = local_b[:, d:] # local_b.shape = batch_size x 100

            h.append((local_f+local_b).unsqueeze(1))
        h = torch.cat(h, dim=1) # h.shape = batch_size x (sentence_max_size-m+1) x 100
        h = self.dropout(h)

        x = self.text_cnn(x.unsqueeze(1)).squeeze().permute(0, 2, 1) # x.shape = batch_size x (sentence_max_size-m+1) x 100
        x = self.dropout(x)
        
        x = x + h

        x, _ = self.lstm(x) # x.shape = batch_size x (sentence_max_size-m+1) x 100
        x = x[:,-1,:].squeeze() # x.shape = batch_size x 100
        x = self.dropout(x)

        img = self.alexnet(img)
        img = self.dropout(img)
        img = self.img_linear(img)
        img = self.dropout(img)

        h = torch.cat([x, img], 1) # b x 200
        
        sen = self.sen_linear(h)
        sen = self.dropout(sen)

        emo = self.emo_linear(h)
        emo = self.dropout(emo)

        return sen, emo