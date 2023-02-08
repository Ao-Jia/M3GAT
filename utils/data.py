from torch.utils.data import Dataset
from utils.load import load

class dataset(Dataset):

    def __init__(self, dataset, mode):
        assert mode in ['train', 'dev', 'test']
        self.dataset = dataset
        self.mode = mode
        self.dialogue_adjs, self.dialogue_tokens, self.dialogue_sen, self.dialogue_emo, self.dialogue_imgs, self.dialogue_sp_idx, self.dialogue_idx = load(self.dataset, self.mode)

    def __getitem__(self, item):
        return self.dialogue_adjs[item], self.dialogue_tokens[item], self.dialogue_sen[item], self.dialogue_emo[item], self.dialogue_imgs[item], self.dialogue_sp_idx[item], self.dialogue_idx[item]

    def __len__(self):
        return len(self.dialogue_sen)