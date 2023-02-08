import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AlbertModel
from torchvision.models import vgg16

class UPBMTL(nn.Module):
    def __init__(self, args):
        super(UPBMTL, self).__init__()

        self.albert = AlbertModel.from_pretrained("albert-base-v2")

        self.vgg = vgg16(pretrained=True)

        self.sen_linear1 = nn.Linear(1768, 512)
        self.sen_linear2 = nn.Linear(512, 256)
        self.sen_linear3 = nn.Linear(256, args.sen_class)

        self.emo_linear1 = nn.Linear(1768, 512)
        self.emo_linear2 = nn.Linear(512, 256)
        self.emo_linear3 = nn.Linear(256, args.emo_class)

        self.dropout = args.dropout

    def forward(self, x, masks, imgs):
        x = self.albert(input_ids=x, attention_mask=masks).pooler_output
        x = F.dropout(x, 0.1, training=self.training)

        img = self.vgg(imgs)

        x = torch.cat([x, img], dim=1)

        x_sen = F.dropout(F.relu(self.sen_linear1(x)), 0.3, training=self.training)
        x_sen = F.dropout(F.relu(self.sen_linear2(x_sen)), 0.3, training=self.training)
        x_sen = self.sen_linear3(x_sen)

        x_emo = F.dropout(F.relu(self.emo_linear1(x)), 0.3, training=self.training)
        x_emo = F.dropout(F.relu(self.emo_linear2(x_emo)), 0.3, training=self.training)
        x_emo = self.emo_linear3(x_emo)

        return x_sen, x_emo