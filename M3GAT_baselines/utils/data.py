from torch.utils.data import Dataset
from utils.load import load

import torch
import numpy as np

class dataset(Dataset):

    def __init__(self, args, embedding, mode):
        assert mode in ['train', 'dev', 'test']
        self.sentence_max_size = args.sentence_max_size
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.mode = mode
        self.embedding = embedding

        if self.model_name == 'MMCNN' or self.model_name == 'BiGRU':
            self.dialogue_tokens, self.dialogue_sen, self.dialogue_emo, self.dialogue_imgs = load(self.model_name, self.dataset, self.mode)
        elif self.model_name == 'UPBMTL':
            self.dialogue_tokens, self.dialogue_masks, self.dialogue_sen, self.dialogue_emo, self.dialogue_imgs = load(self.model_name, self.dataset, self.mode)
        elif self.model_name == 'RoBERTa':
            self.dialogue_tokens, self.dialogue_masks, self.dialogue_sen, self.dialogue_emo, self.dialogue_imgs = load(self.model_name, self.dataset, self.mode)
        elif self.model_name == 'EfficientNet':
            self.dialogue_tokens, self.dialogue_masks, self.dialogue_sen, self.dialogue_emo, self.dialogue_imgs = load(self.model_name, self.dataset, self.mode)
        elif self.model_name == 'mmbt':
            self.dialogue_tokens, self.dialogue_masks, self.dialogue_seg, self.dialogue_sen, self.dialogue_emo, self.dialogue_imgs = load(self.model_name, self.dataset, self.mode)

    def __getitem__(self, idx):
        if self.model_name == 'MMCNN' or self.model_name == 'BiGRU':
            words = []
            for w in self.dialogue_tokens[idx]:
                if w != ' ':
                    if w in self.embedding.keys():
                        words.append(self.embedding[w])
                    else:
                        words.append(self.embedding['<UNK>'])
            if len(words) > self.sentence_max_size:
                words = words[0:self.sentence_max_size]
            else:
                while len(words) < self.sentence_max_size:
                    words.append([0 for _ in range(100)])

            img = self.dialogue_imgs[idx]

            return np.array(words), self.dialogue_sen[idx], self.dialogue_emo[idx], img
            # words.shape = sentence_max_size x 100
        
        elif self.model_name == 'UPBMTL':
            return self.dialogue_tokens[idx], self.dialogue_masks[idx], self.dialogue_sen[idx], self.dialogue_emo[idx], self.dialogue_imgs[idx]
        
        elif self.model_name == 'RoBERTa':
            return self.dialogue_tokens[idx], self.dialogue_masks[idx], self.dialogue_sen[idx], self.dialogue_emo[idx], self.dialogue_imgs[idx]
        
        elif self.model_name == 'EfficientNet':
            return self.dialogue_tokens[idx], self.dialogue_masks[idx], self.dialogue_sen[idx], self.dialogue_emo[idx], self.dialogue_imgs[idx]
        
        elif self.model_name == 'mmbt':
            return self.dialogue_tokens[idx], self.dialogue_masks[idx], self.dialogue_seg[idx], self.dialogue_sen[idx], self.dialogue_emo[idx], self.dialogue_imgs[idx]

    def __len__(self):
        return len(self.dialogue_sen)