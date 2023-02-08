import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer, SpGraphAttentionLayer
from utils.load import expand_list
from transformers import BertModel
from torch.autograd import Variable
import time
import numpy as np
import torchvision.models as models

class M3GAT(nn.Module):
    def __init__(self, dropout, hidden_dim, alpha, nheads, nlayers_TE, nlayers_IE, nlayers_MTI, z, sen_class, emo_class, dataset, mode, task):
        super(M3GAT, self).__init__()

        self.dataset = dataset
        self.mode = mode
        self.task = task
        self.sen_class = sen_class
        self.emo_class = emo_class
        self.hidden_dim = hidden_dim
        if dataset != 'MSED':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_encoder = TextEncoder(nfeat=768, nhid=768, nclass=hidden_dim, dropout=dropout, alpha=alpha, nheads=nheads, nlayers=nlayers_TE, dataset=dataset)
        self.img_encoder = ImageEncoder(nfeat=3, nhid=3, nclass=hidden_dim, dropout=dropout, alpha=alpha, nheads=nheads, nlayers=nlayers_IE, dataset=dataset)
        self.mti_graph = MultiTaskInteractiveGraph(nfeat=hidden_dim, nhid=hidden_dim, nclass=hidden_dim, dropout=dropout, alpha=alpha, nheads=nheads, nlayers=nlayers_MTI, z=z, dataset=dataset, mode=mode, task=task)
        self.linear0 = nn.Linear(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, sen_class)
        self.linear2 = nn.Linear(hidden_dim, emo_class)
        self.linear3 = nn.Linear(768, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim//2, batch_first=True, bidirectional=True)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(dropout)

        if dataset == 'MSED':
            if task != 'e':
                if mode == 't+v':
                    self.linear_sen = nn.Linear(768*2, sen_class)
                else:
                    self.linear_sen = nn.Linear(768, sen_class)
            if task != 's':
                if mode == 't+v':
                    self.linear_emo = nn.Linear(768*2, emo_class)
                else:
                    self.linear_emo = nn.Linear(768, emo_class)

    def padding(self, dialogue_adjs, dialogue_tokens):
        dialogue_tokens_len = len(dialogue_tokens)
        dialogue_tokens_len_list = [len(_) for _ in dialogue_tokens] # the number of utterance in each dialogue
        max_dialogue_tokens_len = max(dialogue_tokens_len_list)

        dialogue_adjs_len = len(dialogue_adjs)
        dialogue_adjs_len_list = [len(_) for _ in dialogue_adjs]
        max_dialogue_adjs_len = max(dialogue_adjs_len_list)

        assert dialogue_adjs_len == dialogue_tokens_len
        assert max_dialogue_adjs_len == max_dialogue_tokens_len
        dialogue_len = dialogue_adjs_len
        max_dialogue_len = max_dialogue_adjs_len

        uttr_tokens_len = []
        for uttr_tokens in dialogue_tokens:
            uttr_tokens_len.extend([len(_) for _ in uttr_tokens]) # the number of tokens in each utterance
        max_uttr_tokens_len = max(uttr_tokens_len)

        uttr_adjs_len = []
        for uttr_adjs in dialogue_adjs:
            uttr_adjs_len.extend([len(_) for _ in uttr_adjs])
        max_uttr_adjs_len = max(uttr_adjs_len)

        assert max_uttr_adjs_len == max_uttr_tokens_len - 2 # [CLS] & [SEP]

        masks = []
        seq_lengths = [] # num of tokens in each utterance after padding
        for dialogue_idx in range(dialogue_len):
            uttr_tokens_len = dialogue_tokens_len_list[dialogue_idx]
            for uttr_idx in range(0, uttr_tokens_len):
                t = len(dialogue_tokens[dialogue_idx][uttr_idx])
                dialogue_tokens[dialogue_idx][uttr_idx].extend([0] * (max_uttr_tokens_len - t))
                masks.append([1] * t + [0] * (max_uttr_tokens_len - t))
                seq_lengths.append(t-2) # [CLS] & [SEP]
            for _ in range(uttr_tokens_len, max_dialogue_len):
                dialogue_tokens[dialogue_idx].append([0] * max_uttr_tokens_len)
                masks.append([0] * max_uttr_tokens_len)
                seq_lengths.append(1)
            
            uttr_adjs_len = dialogue_adjs_len_list[dialogue_idx]
            for uttr_idx in range(0, uttr_adjs_len):
                adj_dim = len(dialogue_adjs[dialogue_idx][uttr_idx])
                for row in dialogue_adjs[dialogue_idx][uttr_idx]:
                    row.extend([0] * (max_uttr_adjs_len - adj_dim))
                for _ in range(adj_dim, max_uttr_adjs_len):
                    dialogue_adjs[dialogue_idx][uttr_idx].append([0] * max_uttr_adjs_len)
            for _ in range(uttr_adjs_len, max_dialogue_len):
                dialogue_adjs[dialogue_idx].append([[0 for _ in range(max_uttr_adjs_len)] for _ in range(max_uttr_adjs_len)])

        masks = torch.LongTensor(masks)
        seq_lengths = torch.LongTensor(seq_lengths)

        return dialogue_adjs, dialogue_tokens, masks, dialogue_tokens_len_list, seq_lengths

    def flat(self, dialogue_imgs):
        if self.dataset == 'MELD':
            new_dialogue_imgs = []
            for i in range(3):
                for dia in dialogue_imgs:
                    for utt in dia[i]:
                        new_dialogue_imgs.append(utt.unsqueeze(0))
            return torch.cat(new_dialogue_imgs, dim=0)
        elif self.dataset == 'MSED':
            new_dialogue_imgs = []
            for dia in dialogue_imgs:
                new_dialogue_imgs.append(dia[0].unsqueeze(0))
            return torch.cat(new_dialogue_imgs, dim=0)

    def forward(self, dialogue_adjs, dialogue_emb, dialogue_imgs, img_adj, utt_num_list, seq_lengths, dialogue_sp_idx, job_embedding, sex_embedding, personality_embedding, sp_jsp_list):
        # dialogue_adjs.shape = d x u x t x t
        # dialogue_emb.shape = d x u x t x 768

        t_sen_features, t_emo_features = self.text_encoder(dialogue_emb, dialogue_adjs, utt_num_list) # t_sen_features.shape = total_utt_num x hidden_dim, t_emo_features.shape = total_utt_num x hidden_dim

        if self.dataset != 'MSED':
            input_sen = []
            for i, utt_num in enumerate(utt_num_list):
                input_sen.append(t_sen_features[i, 0:utt_num, :])
            input_sen = torch.cat(input_sen, dim=0)

            input_emo = []
            for i, utt_num in enumerate(utt_num_list):
                input_emo.append(t_emo_features[i, 0:utt_num, :])
            input_emo = torch.cat(input_emo, dim=0)

        v_sen_features, v_emo_features = self.img_encoder(dialogue_imgs, img_adj) # v_sen_features.shape = total_utt_num x hidden_dim, v_emo_features.shape = total_utt_num x hidden_dim

        if self.dataset == 'MSED':
            if self.mode == 't':
                input_sen = t_sen_features
                input_emo = t_emo_features
            elif self.mode == 'v':
                input_sen = v_sen_features
                input_emo = v_emo_features
            else:
                input_sen = torch.cat([t_sen_features, v_sen_features], dim=-1)
                input_emo = torch.cat([t_emo_features, v_emo_features], dim=-1)

        if self.hidden_dim != 768:
            sen, emo = self.mti_graph(t_sen_features, t_emo_features, v_sen_features, v_emo_features, utt_num_list, dialogue_sp_idx, self.linear3(job_embedding), self.linear3(sex_embedding), self.linear3(personality_embedding), sp_jsp_list) # sen.shape = total_utt_num x hidden_dim, emo.shape = total_utt_num x hidden_dim
        else:
            sen, emo = self.mti_graph(t_sen_features, t_emo_features, v_sen_features, v_emo_features, utt_num_list, dialogue_sp_idx, job_embedding, sex_embedding, personality_embedding, sp_jsp_list) # sen.shape = total_utt_num x hidden_dim, emo.shape = total_utt_num x hidden_dim

        if self.dataset == 'MSED':
            if self.task == 's':
                return self.dropout(self.linear_sen(sen + input_sen)), None
            elif self.task == 'e':
                return None, self.dropout(self.linear_emo(emo + input_emo))
            else:
                return self.dropout(self.linear_sen(sen + input_sen)), self.dropout(self.linear_emo(emo + input_emo))

        sen_prime = self.dropout((self.linear0(sen)))

        start = 0
        emo_prime = []
        for utt_num in utt_num_list:
            h = emo[start:start+utt_num].unsqueeze(0)
            start += utt_num
            h, _ = self.lstm(h)
            emo_prime.append(h.squeeze(0))

        # residual
        sen_res = sen_prime + input_sen
        emo_res = torch.cat(emo_prime, dim=0) + input_emo

        return self.dropout(self.linear1(sen_res)), self.dropout(self.linear2(emo_res))

    def measure(self, dialogue_adjs, dialogue_tokens, dialogue_sen, dialogue_emo, dialogue_imgs, dialogue_sp_idx, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj, predic=False):
        dialogue_adjs, dialogue_tokens, masks, utt_num_list, seq_lengths = self.padding(dialogue_adjs, dialogue_tokens)
        dialogue_imgs = self.flat(dialogue_imgs)  # 3*total_utt_num x 720 x 1280 x 3
        dialogue_tokens = torch.LongTensor(dialogue_tokens)
        dialogue_adjs = torch.FloatTensor(dialogue_adjs)
        dialogue_sen = torch.LongTensor(expand_list(dialogue_sen))
        dialogue_emo = torch.LongTensor(expand_list(dialogue_emo))
        job_embedding = torch.from_numpy(job_embedding)
        sex_embedding = torch.from_numpy(sex_embedding)
        personality_embedding = torch.from_numpy(personality_embedding)

        if torch.cuda.is_available():
            dialogue_adjs = dialogue_adjs.cuda()
            dialogue_tokens = dialogue_tokens.cuda()
            dialogue_sen = dialogue_sen.cuda()
            dialogue_emo = dialogue_emo.cuda()
            masks = masks.cuda()
            dialogue_imgs = dialogue_imgs.cuda()
            job_embedding = job_embedding.cuda()
            sex_embedding = sex_embedding.cuda()
            personality_embedding = personality_embedding.cuda()
        
        dialog_num = len(dialogue_tokens)
        tokens_num = len(dialogue_tokens[0][0])
        dialogue_tokens = dialogue_tokens.view(-1, tokens_num)

        if self.dataset != 'MSED':
            dialogue_features = self.bert(input_ids=dialogue_tokens, attention_mask=masks).last_hidden_state
            dialogue_features = dialogue_features.view(dialog_num, -1, tokens_num, 768)[:, :, 1:-1, :] # delete [CLS] & [SEP]
        else:
            dialogue_features = [dialogue_tokens, masks]

        sen, emo = self.forward(dialogue_adjs, dialogue_features, dialogue_imgs, img_adj, utt_num_list, seq_lengths, dialogue_sp_idx, job_embedding, sex_embedding, personality_embedding, sp_jsp_list)

        if predic:
            return sen, emo
        else:
            if self.task == 's':
                sen_loss = self.criterion(sen, dialogue_sen)
                return sen_loss
            elif self.task == 'e':
                emo_loss = self.criterion(emo, dialogue_emo)
                return emo_loss
            else:
                sen_loss = self.criterion(sen, dialogue_sen)
                emo_loss = self.criterion(emo, dialogue_emo)
                return sen_loss + emo_loss


class TextEncoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers, dataset):
        super(TextEncoder, self).__init__()

        self.dataset = dataset
        self.nclass = nclass
        self.dropout = nn.Dropout(dropout)

        if dataset != 'MSED':
            self.GATs = [GAT(nfeat, nhid, nhid, dropout, alpha, nheads) for _ in range(nlayers)]
            for i, _GAT in enumerate(self.GATs):
                self.add_module('GAT_{}'.format(i), _GAT)
            self.BiLSTM = nn.LSTM(input_size=nhid, hidden_size=nhid, batch_first=True, dropout=dropout, bidirectional=True)
            self.sen = nn.Linear(nhid*2, nclass)
            self.emo = nn.Linear(nhid*2, nclass)

        else:
            self.bert_sen = BertModel.from_pretrained('bert-base-uncased')
            self.bert_emo = BertModel.from_pretrained('bert-base-uncased')
            if nclass != 768:
                self.sen = nn.Linear(768, nclass)
                self.emo = nn.Linear(768, nclass)

    def forward(self, x, adj, seq_lengths):
        if self.dataset != 'MSED':
            # x.shape = d x u x t x 768
            input_x = x
            x = self.dropout(x)

            for  _GAT in self.GATs:
                x = _GAT(x, adj) # x.shape = d x u x t x 768

            x = input_x + x

            d_num = x.shape[0]
            u_num = x.shape[1]
            t_num = x.shape[-2]
            x = x.view(-1, t_num, 768)
            batch_size = x.shape[0]

            h, _ = self.BiLSTM(x) # h.shape = batch_size x t_num x nhid*2
            h = self.bi_fetch(h, seq_lengths, batch_size, t_num) # h.shape = batch_size x nhid*2
            h = h.view(d_num, u_num, -1) # h_sen.shape = d x u x nhid*2

            return self.dropout(self.sen(h)), self.dropout(self.emo(h))
        
        else:
            # x.shape = d x 768
            x_sen = self.bert_sen(input_ids=x[0], attention_mask=x[1]).pooler_output
            x_emo = self.bert_emo(input_ids=x[0], attention_mask=x[1]).pooler_output

            if self.nclass == 768:
                return x_sen, x_emo
            else:
                return self.dropout(self.sen(x_sen)), self.dropout(self.emo(x_emo))


    
    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        if torch.cuda.is_available():
            seq_lengths = seq_lengths.cuda()
            fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
            fw_out = fw_out.view(batch_size * max_len, -1)
            bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
            bw_out = bw_out.view(batch_size * max_len, -1)

            batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
            batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()
        else:
            fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])))
            fw_out = fw_out.view(batch_size * max_len, -1)
            bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])))
            bw_out = bw_out.view(batch_size * max_len, -1)

            batch_range = Variable(torch.LongTensor(range(batch_size))) * max_len
            batch_zeros = Variable(torch.zeros(batch_size).long())

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = self.dropout(x)
        x = F.elu(self.out_att(x, adj), inplace=True)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers, dataset):
        super(ImageEncoder, self).__init__()

        self.dataset = dataset
        self.nclass = nclass

        if dataset != 'MSED':
            self.resnet18 = models.resnet18(pretrained=True)
        else:
            self.resnet_sen = models.resnet18(pretrained=True)
            self.resnet_emo = models.resnet18(pretrained=True)
        if dataset == 'MELD':
            self.sen = nn.Linear(3000, nclass)
            self.emo = nn.Linear(3000, nclass)
        else:
            self.sen = nn.Linear(1000, nclass)
            self.emo = nn.Linear(1000, nclass)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        if self.dataset != 'MSED':
            x = x.permute(0, 3, 1, 2)
            out = self.resnet18(x) # out.shape = 3*total_utt_num x 1000
            del x

            total_utt_num = out.shape[0] // 3
            out1 = torch.cat([out[0:total_utt_num, :], out[total_utt_num:2*total_utt_num, :], out[2*total_utt_num:3*total_utt_num, :]], dim=-1) # total_utt_num x 3000
            return self.dropout(self.sen(out1)), self.dropout(self.emo(out1)) # total_utt_num x nclass
        
        else:
            x = x.permute(0, 3, 1, 2)
            sen = self.resnet_sen(x) # out.shape = total_utt_num x 1000
            emo = self.resnet_emo(x)
            del x

            return self.dropout(self.sen(sen)), self.dropout(self.emo(emo)) # total_utt_num x nclass

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.cat([torch.cat([att(row, adj).unsqueeze(0) for row in x], dim=0) for att in self.attentions], dim=-1)

        x = self.dropout(x)
        x = F.elu(torch.cat([self.out_att(row, adj).unsqueeze(0) for row in x], dim=0), inplace=True) # x.shape = 3*total_utt_num x (720*1280) x 3
        x = self.dropout(x)

        return x


class MultiTaskInteractiveGraph(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers, z, dataset, mode, task):
        super(MultiTaskInteractiveGraph, self).__init__()

        self.dataset = dataset
        self.mode = mode
        self.task = task
        self.nclass = nclass
        self.dropout = nn.Dropout(dropout)
        self.z = z

        self.ConGATs = [ConGAT(nfeat, nhid, nclass, dropout, alpha, nheads) for _ in range(nlayers)]
        for i, _ConGAT in enumerate(self.ConGATs):
            self.add_module('ConGAT_{}'.format(i), _ConGAT)

    def forward(self, t_sen_features, t_emo_features, v_sen_features, v_emo_features, utt_num_list, dialogue_sp_idx, job_embedding, sex_embedding, personality_embedding, sp_jsp_list):
        if self.dataset != 'MSED':
            # t_sen_features.shape = d x u x hidden_dim, t_emo_features.shape = d x u x hidden_dim
            # v_sen_features.shape = total_utt_num x hidden_dim, v_emo_features.shape = total_utt_num x hidden_dim

            d = t_sen_features.shape[0]
            v_start = 0
            v_end = 0

            sen = []
            emo = []
            for dialogue_idx in range(d):
                sp_idx = dialogue_sp_idx[dialogue_idx]
                sp_num = len(set(sp_idx))
                utt_sp_idx = [list(set(sp_idx)).index(idx) for idx in sp_idx]
                
                sp_info = []
                for sp in set(sp_idx):
                    jsp = sp_jsp_list[sp]
                    job = job_embedding[jsp[0]]
                    sex = sex_embedding[jsp[1]]
                    personality = personality_embedding[jsp[2]]
                    speaker = torch.mean(torch.cat([job.unsqueeze(0), sex.unsqueeze(0), personality.unsqueeze(0)], dim=0), dim=0)
                    sp_info.append(torch.cat([speaker.unsqueeze(0), job.unsqueeze(0), sex.unsqueeze(0), personality.unsqueeze(0)], dim=0)) # 4 x hidden_dim
                sp_info = torch.cat(sp_info, dim=0) # (4*sp_num) x hidden_dim

                utt_num = utt_num_list[dialogue_idx]
                v_end += utt_num

                t_sen = t_sen_features[dialogue_idx][0:utt_num] # utt_num x hidden_dim
                t_emo = t_emo_features[dialogue_idx][0:utt_num] # utt_num x hidden_dim
                v_sen = v_sen_features[v_start:v_end] # utt_num x hidden_dim
                v_emo = v_emo_features[v_start:v_end] # utt_num x hidden_dim

                v_start = v_end

                adj = self.gen_adj(utt_num, sp_num, utt_sp_idx, self.z)
                if torch.cuda.is_available():
                    adj = adj.cuda()

                x = torch.cat([t_sen, t_emo, v_sen, v_emo, sp_info, sp_info, sp_info, sp_info], dim=0) # (4*utt_num + 16*sp_num) x hidden_dim
                for  _ConGAT in self.ConGATs:
                    x = _ConGAT(x, adj)

                t_sen = x[0:utt_num]
                t_emo = x[utt_num:2*utt_num]
                v_sen = x[2*utt_num:3*utt_num]
                v_emo = x[3*utt_num:4*utt_num]
                del x
                del adj

                sen.append(t_sen)
                emo.append(t_emo)
                del t_sen
                del v_sen
                del t_emo
                del v_emo
            
            return torch.cat(sen, dim=0), torch.cat(emo, dim=0)
        
        else:
            # t_sen_features.shape = d x hidden_dim, t_emo_features.shape = d x hidden_dim
            # v_sen_features.shape = d x hidden_dim, v_emo_features.shape = d x hidden_dim

            d = t_sen_features.shape[0]
            
            if self.mode == 't':
                if self.task == 's':
                    x = t_sen_features
                elif self.task == 'e':
                    x = t_emo_features
                else:
                    x = torch.cat([t_sen_features, t_emo_features], dim=0)
            elif self.mode == 'v':
                if self.task == 's':
                    x = v_sen_features
                elif self.task == 'e':
                    x = v_emo_features
                else:
                    x = torch.cat([v_sen_features, v_emo_features], dim=0)
            else:
                if self.task == 's':
                    x = torch.cat([t_sen_features, v_sen_features], dim=0)
                elif self.task == 'e':
                    x = torch.cat([t_emo_features, v_emo_features], dim=0)
                else:
                    x = torch.cat([t_sen_features, t_emo_features, v_sen_features, v_emo_features], dim=0)
            adj = self.gen_adj(d, None, None)

            if torch.cuda.is_available():
                adj = adj.cuda()

            for  _ConGAT in self.ConGATs:
                x = _ConGAT(x, adj)

            if self.mode == 't':
                if self.task == 's':
                    return x, None
                elif self.task == 'e':
                    return None, x
                else:
                    t_sen, t_emo = torch.chunk(x, 2, dim=0)
                    return t_sen, t_emo
            elif self.mode == 'v':
                if self.task == 's':
                    return x, None
                elif self.task == 'e':
                    return None, x
                else:
                    v_sen, v_emo = torch.chunk(x, 2, dim=0)
                    return v_sen, v_emo
            else:
                if self.task == 's':
                    t_sen, v_sen = torch.chunk(x, 2, dim=0)
                    return torch.cat([t_sen, v_sen], dim=-1), None
                elif self.task == 'e':
                    t_emo, v_emo = torch.chunk(x, 2, dim=0)
                    return None, torch.cat([t_emo, v_emo], dim=-1)
                else:
                    t_sen, t_emo, v_sen, v_emo = torch.chunk(x, 4, dim=0)
                    
                    return torch.cat([t_sen, v_sen], dim=1), torch.cat([t_emo, v_emo], dim=1)
            
    def gen_adj(self, utt_num, sp_num, utt_sp_idx):
        if self.dataset != 'MSED':
            adj = torch.zeros((utt_num, utt_num), dtype=int)
            for i, row in enumerate(adj):
                for j, col in enumerate(row):
                    if abs(i-j) <= self.z:
                        adj[i][j] = 1
            adj1 = torch.cat([adj, adj, adj, torch.zeros_like(adj)], dim=0)
            adj2 = torch.cat([adj, adj, torch.zeros_like(adj), adj], dim=0)
            adj3 = torch.cat([adj, torch.zeros_like(adj), adj, adj], dim=0)
            adj4 = torch.cat([torch.zeros_like(adj), adj, adj, adj], dim=0)
            adj = torch.cat([adj1, adj2, adj3, adj4], dim=1)
            cen = utt_num // 2
            for i in range(4):
                c = cen + i * utt_num
                for j in range(utt_num):
                    adj[c][i*utt_num+j] = 1

            adj = torch.cat([torch.cat([adj, torch.zeros((4*utt_num, 16*sp_num), dtype=int)], dim=1), torch.zeros((16*sp_num, 4*utt_num+16*sp_num), dtype=int)], dim=0)
            # connect to speaker node
            for row in range(4*utt_num):
                utt_idx = row % utt_num
                sp_idx = utt_sp_idx[utt_idx]
                adj[row][4*utt_num + 4*sp_num*(row // utt_num) + 4*sp_idx] = 1
                adj[4*utt_num + 4*sp_num*(row // utt_num) + 4*sp_idx][row] = 1
            # speaker, job, sex, personality node connect to each other
            for idx in range(4*sp_num):
                row = 4 * utt_num + 4 * idx
                for i in range(4):
                    for j in range(4):
                        adj[row+i][row+j] = 1
            
            return adj
        
        else:
            if self.mode == 't':
                if self.task == 's':
                    adj = torch.eye(utt_num, dtype=int)
                elif self.task == 'e':
                    adj = torch.eye(utt_num, dtype=int)
                else:
                    adj1 = torch.cat([torch.eye(utt_num, dtype=int), torch.eye(utt_num, dtype=int)], dim=0)
                    adj = torch.cat([adj1, adj1], dim=1)
            elif self.mode == 'v':
                if self.task == 's':
                    adj = torch.eye(utt_num, dtype=int)
                elif self.task == 'e':
                    adj = torch.eye(utt_num, dtype=int)
                else:
                    adj1 = torch.cat([torch.eye(utt_num, dtype=int), torch.eye(utt_num, dtype=int)], dim=0)
                    adj = torch.cat([adj1, adj1], dim=1)
            else:
                if self.task == 's':
                    adj1 = torch.cat([torch.eye(utt_num, dtype=int), torch.eye(utt_num, dtype=int)], dim=0)
                    adj = torch.cat([adj1, adj1], dim=1)
                elif self.task == 'e':
                    adj1 = torch.cat([torch.eye(utt_num, dtype=int), torch.eye(utt_num, dtype=int)], dim=0)
                    adj = torch.cat([adj1, adj1], dim=1)
                else:
                    adj = torch.eye(utt_num, dtype=int)
                    
                    adj1 = torch.cat([adj, adj, adj, torch.zeros_like(adj)], dim=0)
                    adj2 = torch.cat([adj, adj, torch.zeros_like(adj), adj], dim=0)
                    adj3 = torch.cat([adj, torch.zeros_like(adj), adj, adj], dim=0)
                    adj4 = torch.cat([torch.zeros_like(adj), adj, adj, adj], dim=0)

                    # # no cross-modal
                    # adj1 = torch.cat([adj, adj, torch.zeros_like(adj), torch.zeros_like(adj)], dim=0)
                    # adj2 = torch.cat([adj, adj, torch.zeros_like(adj), torch.zeros_like(adj)], dim=0)
                    # adj3 = torch.cat([torch.zeros_like(adj), torch.zeros_like(adj), adj, adj], dim=0)
                    # adj4 = torch.cat([torch.zeros_like(adj), torch.zeros_like(adj), adj, adj], dim=0)

                    # # no cross task
                    # adj1 = torch.cat([adj, torch.zeros_like(adj), adj, torch.zeros_like(adj)], dim=0)
                    # adj2 = torch.cat([torch.zeros_like(adj), adj, torch.zeros_like(adj), adj], dim=0)
                    # adj3 = torch.cat([adj, torch.zeros_like(adj), adj, torch.zeros_like(adj)], dim=0)
                    # adj4 = torch.cat([torch.zeros_like(adj), adj, torch.zeros_like(adj), adj], dim=0)

                    adj = torch.cat([adj1, adj2, adj3, adj4], dim=1)

            return adj


class ConGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(ConGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        input_x = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = self.dropout(x)
        x = F.elu(self.out_att(x, adj), inplace=True)

        return input_x + x