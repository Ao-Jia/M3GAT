import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, args):
        super(EfficientNet, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        self.sen_linear = nn.Linear(1768, args.sen_class)
        self.emo_linear = nn.Linear(1768, args.emo_class)

        self.dropout = 0.1

    def forward(self, x, masks, imgs):
        x = x.squeeze()
        masks = masks.squeeze()
        x = self.bert(x, attention_mask=masks).pooler_output

        imgs = F.dropout(self.efficientnet(imgs), self.dropout, training=self.training)

        x = torch.cat([x, imgs], dim=1)

        sen = self.sen_linear(x)
        emo = self.emo_linear(x)

        return sen, emo