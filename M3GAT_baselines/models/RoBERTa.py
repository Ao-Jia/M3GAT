import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import RobertaModel
from torchvision.models import vgg16


class RoBERTa(nn.Module):
    def __init__(self, args):
        super(RoBERTa, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.gru = nn.GRU(input_size=768,
                          hidden_size=768,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False)

        self.sen_linear1 = nn.Linear(768, 512)
        self.sen_linear2 = nn.Linear(512, 256)
        self.sen_linear3 = nn.Linear(256, args.sen_class)

        self.emo_linear1 = nn.Linear(768, 512)
        self.emo_linear2 = nn.Linear(512, 256)
        self.emo_linear3 = nn.Linear(256, args.emo_class)

        self.dropout = args.dropout

    def forward(self, x, masks, imgs):
        x = x.squeeze()
        masks = masks.squeeze()
        x = self.roberta(input_ids=x, attention_mask=masks).last_hidden_state

        x, _ = self.gru(x) # b x L x hidden_dim

        x = x[:, -1, :] # b x hidden_dim

        x_sen = self.sen_linear1(x)
        x_sen = self.sen_linear2(x_sen)
        x_sen = F.dropout(self.sen_linear3(x_sen), 0.5, training=self.training)

        x_emo = self.emo_linear1(x)
        x_emo = self.emo_linear2(x_emo)
        x_emo = F.dropout(self.emo_linear3(x_emo), 0.5, training=self.training)

        return x_sen, x_emo