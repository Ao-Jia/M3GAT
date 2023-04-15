import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import spacy
import csv
from transformers import BertTokenizer
import os
from PIL import Image
import torchvision

def load(dataset, mode):
    assert mode in ['train', 'dev', 'test']

    if dataset == 'MELD':
        csv_path = os.path.join('./data/MELD', '{}_sent_emo.csv'.format(mode))
        img_dir = './data/MELD/{}_img_small'.format(mode)
        return read_data(csv_path, img_dir, dataset, mode)
    
    elif dataset == 'MEISD':
        csv_path = './data/MEISD/MEISD_text.csv'
        img_dir = './data/MEISD/VIDEO FEATURES'
        return read_data(csv_path, img_dir, dataset, mode)

    else: # dataset == 'MSED'
        csv_path = './data/MSED/{}/{}.csv'.format(mode, mode)
        img_dir = './data/MSED/{}/images'.format(mode)
        return read_data(csv_path, img_dir, dataset, mode)


def read_data(csv_path, img_dir, dataset, mode):
    if dataset == 'MELD':
        dialogue_adjs = []
        dialogue_tokens = []
        dialogue_sen = []
        dialogue_emo = []
        dialogue_imgs = []
        dialogue_sp_idx = []

        nlp = spacy.load('en_core_web_sm')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        trans = torchvision.transforms.Compose([torchvision.transforms.Resize((180, 320)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        sen_list = ['positive', 'neutral', 'negative']
        emo_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        names = get_speaker_name()
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                fieldnames = next(reader)
                csv_reader = csv.DictReader(f, fieldnames=fieldnames)
                dialogue_id = 0
                uttr_adjs = []
                uttr_tokens = []
                uttr_sen = []
                uttr_emo = []
                uttr_imgs = [[],[],[]]
                uttr_sp_idx = []

                for row in csv_reader:
                    if row['Dialogue_ID'] != str(dialogue_id):
                        dialogue_adjs.append(uttr_adjs)
                        dialogue_tokens.append(uttr_tokens)
                        dialogue_sen.append(uttr_sen)
                        dialogue_emo.append(uttr_emo)
                        dialogue_imgs.append(uttr_imgs)
                        dialogue_sp_idx.append(uttr_sp_idx)

                        uttr_adjs = []
                        uttr_tokens = []
                        uttr_sen = []
                        uttr_emo = []
                        uttr_imgs = [[],[],[]]
                        uttr_sp_idx = []

                        dialogue_id += 1
                    
                    uttr = row['Utterance']
                    doc = nlp(uttr)
                    tokens_ids = tokenizer.convert_tokens_to_ids([token.text.lower() for token in doc])

                    assert len(doc) == len(tokens_ids)

                    tokens_ids.insert(0, 101)
                    tokens_ids.append(102)
                    uttr_tokens.append(tokens_ids)

                    idx_map = {token: i for i, token in enumerate(doc)}
                    edges_unodered = [[token, token.head] for token in doc]
                    edges_unodered_flatten = []
                    for edge in edges_unodered:
                        edges_unodered_flatten.append(edge[0])
                        edges_unodered_flatten.append(edge[1])
                    edges = np.array(list(map(idx_map.get, edges_unodered_flatten)), dtype=np.int32).reshape((len(edges_unodered), 2))
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(doc), len(doc)), dtype=np.float32)

                    # build symmetric adjacency matrix
                    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
                    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
                    adj = np.array(adj.todense()).tolist()
                    uttr_adjs.append(adj)

                    sen = row['Sentiment']
                    uttr_sen.append(sen_list.index(sen))

                    emo = row['Emotion']
                    uttr_emo.append(emo_list.index(emo))

                    for i in range(3):
                        img_name = 'dia{}_utt{}_{}.jpg'.format(row['Dialogue_ID'],row['Utterance_ID'], i)
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        uttr_imgs[i].append(trans(img).permute(1, 2, 0))
                    
                    uttr_sp_idx.append(names.index(row['Speaker']))
                
                dialogue_adjs.append(uttr_adjs)
                dialogue_tokens.append(uttr_tokens)
                dialogue_sen.append(uttr_sen)
                dialogue_emo.append(uttr_emo)
                dialogue_imgs.append(uttr_imgs)
                dialogue_sp_idx.append(uttr_sp_idx)

        return dialogue_adjs, dialogue_tokens, dialogue_sen, dialogue_emo, dialogue_imgs, dialogue_sp_idx
    
    elif dataset == 'MEISD':
        nlp = spacy.load('en_core_web_sm')
        dialogue_tokens = []
        dialogue_adjs = []
        dialogue_sen = []
        dialogue_emo = []
        dialogue_imgs = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        sen_list = ['positive', 'neutral', 'negative']
        emo_list = ['anger', 'disgust', 'joy', 'surprise', 'acceptance', 'sadness', 'fear', 'neutral', 'like']

        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
            csv_reader = csv.DictReader(f, fieldnames=fieldnames)
            if mode == 'dev': # 1-112
                dialogue_id = 1
            elif mode == 'train': # 225-1125
                dialogue_id = 225
            else: # 113-225
                dialogue_id = 113
            uttr_tokens = []
            uttr_adjs = []
            uttr_sen = []
            uttr_emo = []
            uttr_imgs = []

            for idx, row in enumerate(csv_reader):
                dialog_ids = int(row['dialog_ids'])
                if mode == 'dev': # 1-112
                    if dialog_ids > 112:
                        break
                elif mode == 'train': # 225-1125
                    if dialog_ids < 225:
                        continue
                else: # 113-225
                    if dialog_ids < 113:
                        continue
                    if dialog_ids > 225:
                        break
                
                if row['dialog_ids'] != str(dialogue_id):
                    dialogue_tokens.append(uttr_tokens)
                    dialogue_sen.append(uttr_sen)
                    dialogue_emo.append(uttr_emo)
                    dialogue_adjs.append(uttr_adjs)
                    dialogue_imgs.append(uttr_imgs)

                    uttr_tokens = []
                    uttr_adjs = []
                    uttr_sen = []
                    uttr_emo = []
                    uttr_imgs = []

                    dialogue_id += 1
                
                uttr = row['Utterances']
                sen = row['sentiment'].strip().lower()
                emo = row['emotion'].strip().lower()

                if sen not in sen_list:
                    continue
                if emo not in emo_list:
                    continue

                doc = nlp(uttr)
                tokens_ids = tokenizer.convert_tokens_to_ids([token.text.lower() for token in doc])
                tokens_ids.insert(0, 101)
                tokens_ids.append(102)
                uttr_tokens.append(tokens_ids)
                
                idx_map = {token: i for i, token in enumerate(doc)}
                edges_unodered = [[token, token.head] for token in doc]
                edges_unodered_flatten = []
                for edge in edges_unodered:
                    edges_unodered_flatten.append(edge[0])
                    edges_unodered_flatten.append(edge[1])
                edges = np.array(list(map(idx_map.get, edges_unodered_flatten)), dtype=np.int32).reshape((len(edges_unodered), 2))
                adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(doc), len(doc)), dtype=np.float32)
                # build symmetric adjacency matrix
                adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
                adj = normalize_adj(adj + sp.eye(adj.shape[0]))
                adj = np.array(adj.todense()).tolist()
                uttr_adjs.append(adj)

                uttr_sen.append(sen_list.index(sen))
                uttr_emo.append(emo_list.index(emo))

                img_name = '{}_features.npy'.format(idx)
                img_path = os.path.join(img_dir, img_name)
                assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                img = np.load(img_path)
                uttr_imgs.append(img[img.shape[0] // 2].astype('float32')) # 2048
            
            dialogue_tokens.append(uttr_tokens)
            dialogue_sen.append(uttr_sen)
            dialogue_emo.append(uttr_emo)
            dialogue_adjs.append(uttr_adjs)
            dialogue_imgs.append(uttr_imgs)

        return dialogue_adjs, dialogue_tokens, dialogue_sen, dialogue_emo, dialogue_imgs, None

    else: # dataset == 'MSED'
        nlp = spacy.load('en_core_web_sm')
        trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dialogue_tokens = []
        dialogue_adjs = []
        dialogue_sen = []
        dialogue_emo = []
        dialogue_imgs = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        sen_list = ['positive', 'neutral', 'negative']
        emo_list = ['happiness', 'neutral', 'anger', 'disgust', 'fear', 'sad']

        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
            csv_reader = csv.DictReader(f, fieldnames=fieldnames)
            uttr_tokens = []
            uttr_sen = []
            uttr_emo = []
            uttr_adjs = []
            uttr_imgs = []

            for idx, row in enumerate(csv_reader):           
                uttr = row['Caption']
                doc = nlp(uttr)
                tokens_ids = tokenizer.convert_tokens_to_ids([token.text.lower() for token in doc])

                tokens_ids.insert(0, 101)
                tokens_ids.append(102)
                uttr_tokens.append(tokens_ids)

                idx_map = {token: i for i, token in enumerate(doc)}
                edges_unodered = [[token, token.head] for token in doc]
                edges_unodered_flatten = []
                for edge in edges_unodered:
                    edges_unodered_flatten.append(edge[0])
                    edges_unodered_flatten.append(edge[1])
                edges = np.array(list(map(idx_map.get, edges_unodered_flatten)), dtype=np.int32).reshape((len(edges_unodered), 2))
                adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(doc), len(doc)), dtype=np.float32)
                # build symmetric adjacency matrix
                adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
                adj = normalize_adj(adj + sp.eye(adj.shape[0]))
                adj = np.array(adj.todense()).tolist()
                uttr_adjs.append(adj)

                sen = row['Sentiment']
                uttr_sen.append(sen_list.index(sen))

                emo = row['Emotion']
                uttr_emo.append(emo_list.index(emo))

                img_name = '{}.jpg'.format(idx+1)
                img_path = os.path.join(img_dir, img_name)
                assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                img = Image.open(img_path)
                uttr_imgs.append(trans(img).permute(1, 2, 0))
            
                dialogue_tokens.append(uttr_tokens)
                dialogue_sen.append(uttr_sen)
                dialogue_emo.append(uttr_emo)
                dialogue_adjs.append(uttr_adjs)
                dialogue_imgs.append(uttr_imgs)

                uttr_tokens = []
                uttr_sen = []
                uttr_emo = []
                uttr_adjs = []
                uttr_imgs = []

        return dialogue_adjs, dialogue_tokens, dialogue_sen, dialogue_emo, dialogue_imgs, None


def get_speaker_name():
    names = []
    with open('./data/MELD/speaker_information.csv', 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        csv_reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in csv_reader:
            names.append(row['name'])
    return names


def get_jsp_list(): # job, sex, personality
    job = []
    sex = []
    personality = []
    with open('./data/MELD/speaker_information.csv', 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        csv_reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in csv_reader:
            if row['job'] not in job:
                job.append(row['job'])
            if row['sex'] not in sex:
                sex.append(row['sex'])
            if row['personality'] not in personality:
                personality.append(row['personality'])
    return job, sex, personality


def get_jsp_embedding():
    job_embedding = np.loadtxt(open('./data/MELD/job_embedding.txt'), delimiter=" ", skiprows=0, dtype=np.float32)
    sex_embedding = np.loadtxt(open('./data/MELD/sex_embedding.txt'), delimiter=" ", skiprows=0, dtype=np.float32)
    personality_embedding = np.loadtxt(open('./data/MELD/personality_embedding.txt'), delimiter=" ", skiprows=0, dtype=np.float32)
    return job_embedding, sex_embedding, personality_embedding


def get_sp_jsp_list():
    job, sex, personality = get_jsp_list()
    sp_jsp_list = []
    with open('./data/MELD/speaker_information.csv', 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        csv_reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in csv_reader:
            sp_jsp_list.append([job.index(row['job']), sex.index(row['sex']), personality.index(row['personality'])])
    return sp_jsp_list


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def expand_list(n_list):
    e_list = []
    for element in n_list:
        if isinstance(element, (list, tuple)):
            e_list.extend(expand_list(element))
        else:
            e_list.append(element)
    return e_list


# childs will also be connected during the loop
def set_adj(adj, idx_map, i, ancestor, n):
    j = idx_map[ancestor]
    adj[i][j] = 1
    adj[j][i] = 1

    if n <= 1 or ancestor.dep_ == 'ROOT':
        return
    else:
        set_adj(adj, idx_map, i, ancestor.head, n-1)