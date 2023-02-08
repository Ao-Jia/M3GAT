import numpy as np
import torch
import spacy
import csv
from transformers import BertTokenizer, AlbertTokenizer, RobertaTokenizer
import os
from PIL import Image
import torchvision
import jieba

def load(model_name, dataset, mode):
    assert mode in ['train', 'dev', 'test']

    if dataset == 'MELD':
        csv_path = os.path.join('./data/MELD', '{}_sent_emo.csv'.format(mode))
        img_dir = './data/MELD/{}_img_small'.format(mode)
        return read_data(model_name, csv_path, img_dir, dataset)

    elif dataset == 'MSED':
        csv_path = os.path.join('/home/jiaao/MSED/data/MSED', '{}/{}.csv'.format(mode, mode))
        img_dir = os.path.join('/home/jiaao/MSED/data/MSED', '{}/images'.format(mode))
        return read_data(model_name, csv_path, img_dir, dataset)

    else:
        pass


def read_data(model_name, csv_path, img_dir, dataset):
    if dataset == 'MELD':
        if model_name == 'MMCNN' or model_name == 'BiGRU':
            dialogue_tokens = []
            dialogue_sen = []
            dialogue_emo = []
            dialogue_imgs = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for row in csv_reader:                  
                        uttr = row['Utterance']
                        dialogue_tokens.append(jieba.cut(uttr))

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

                        img_name = 'dia{}_utt{}_1.jpg'.format(row['Dialogue_ID'], row['Utterance_ID'])
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        dialogue_imgs.append(trans(img))

            return dialogue_tokens, dialogue_sen, dialogue_emo, dialogue_imgs
        
        elif model_name == 'UPBMTL':
            dialogue_tokens = []
            dialogue_masks = []
            dialogue_sen = []
            dialogue_emo = []
            dialogue_imgs = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", cache_dir='/home/jiaao/data/albert-base-v2')
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for row in csv_reader:                  
                        uttr = row['Utterance']
                        out = tokenizer(uttr, add_special_tokens=True, max_length = 32, padding='max_length', truncation=True, return_tensors='np')
                        dialogue_tokens.append(out.input_ids)
                        dialogue_masks.append(out.attention_mask)

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

                        img_name = 'dia{}_utt{}_1.jpg'.format(row['Dialogue_ID'], row['Utterance_ID'])
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        dialogue_imgs.append(trans(img))

            return dialogue_tokens, dialogue_masks, dialogue_sen, dialogue_emo, dialogue_imgs
            # dialogue_tokens = []
            # dialogue_masks = []
            # dialogue_sen = []
            # dialogue_emo = []
            # dialogue_imgs = []
            # tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", cache_dir='/home/jiaao/data/albert-base-v2')
            # trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            # sen_list = ['positive', 'neutral', 'negative']
            # emo_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

            # with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            #     reader = csv.reader(f)
            #     fieldnames = next(reader)
            #     csv_reader = csv.DictReader(f, fieldnames=fieldnames)
            #     dialogue_id = 0
            #     uttr_tokens = []
            #     uttr_masks = []
            #     uttr_sen = []
            #     uttr_emo = []
            #     uttr_imgs = []

            #     for row in csv_reader:
            #         if row['Dialogue_ID'] != str(dialogue_id):
            #             dialogue_tokens.append(np.vstack(uttr_tokens))
            #             dialogue_masks.append(np.vstack(uttr_masks))
            #             dialogue_sen.append(uttr_sen)
            #             dialogue_emo.append(uttr_emo)
            #             dialogue_imgs.append(torch.concatenate(uttr_imgs, dim=0))

            #             uttr_tokens = []
            #             uttr_masks = []
            #             uttr_sen = []
            #             uttr_emo = []
            #             uttr_imgs = []

            #             dialogue_id += 1

            #         uttr = row['Utterance']
            #         out = tokenizer(uttr, add_special_tokens=True, max_length = 32, padding='max_length', truncation=True, return_tensors='np')
            #         tokens_ids = out.input_ids
            #         masks = out.attention_mask

            #         uttr_tokens.append(tokens_ids)
            #         uttr_masks.append(masks)

            #         sen = row['Sentiment']
            #         uttr_sen.append(sen_list.index(sen))

            #         emo = row['Emotion']
            #         uttr_emo.append(emo_list.index(emo))

            #         img_name = 'dia{}_utt{}_1.jpg'.format(row['Dialogue_ID'], row['Utterance_ID'])
            #         img_path = os.path.join(img_dir, img_name)
            #         assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
            #         img = Image.open(img_path)
            #         uttr_imgs.append(trans(img).unsqueeze(0))

            #     dialogue_tokens.append(np.vstack(uttr_tokens))
            #     dialogue_masks.append(np.vstack(uttr_masks))
            #     dialogue_sen.append(uttr_sen)
            #     dialogue_emo.append(uttr_emo)
            #     dialogue_imgs.append(uttr_imgs)
            
            # return dialogue_tokens, dialogue_masks, dialogue_sen, dialogue_emo, dialogue_imgs
        
        elif model_name == 'RoBERTa':
            dialogue_tokens = []
            dialogue_masks = []
            dialogue_sen = []
            dialogue_emo = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='/home/jiaao/data/roberta-base')

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for row in csv_reader:                  
                        uttr = row['Utterance']
                        out = tokenizer(uttr, add_special_tokens=True, max_length = 32, padding='max_length', truncation=True, return_tensors='np')
                        dialogue_tokens.append(out.input_ids)
                        dialogue_masks.append(out.attention_mask)

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

            return dialogue_tokens, dialogue_masks, dialogue_sen, dialogue_emo
        
        elif model_name == 'EfficientNet':
            dialogue_tokens = []
            dialogue_masks = []
            dialogue_sen = []
            dialogue_emo = []
            dialogue_imgs = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/home/jiaao/data/bert-base-uncased', do_lower_case=True)
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for row in csv_reader:
                        uttr = row['Utterance']
                        out = tokenizer(uttr, add_special_tokens=True, max_length = 32, padding='max_length', truncation=True, return_tensors='np')
                        dialogue_tokens.append(out.input_ids)
                        dialogue_masks.append(out.attention_mask)

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

                        img_name = 'dia{}_utt{}_1.jpg'.format(row['Dialogue_ID'], row['Utterance_ID'])
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        dialogue_imgs.append(trans(img))

            return dialogue_tokens, dialogue_masks, dialogue_sen, dialogue_emo, dialogue_imgs
        
        elif model_name == 'mmbt':
            dialogue_tokens = []
            dialogue_masks = []
            dialogue_seg = []
            dialogue_sen = []
            dialogue_emo = []
            dialogue_imgs = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/home/jiaao/data/bert-base-uncased', do_lower_case=True)
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for row in csv_reader:                  
                        uttr = row['Utterance']
                        out = tokenizer(uttr, add_special_tokens=False, max_length = 32 - 2, padding='max_length', truncation=True, return_tensors='np')
                        dialogue_tokens.append(out.input_ids)
                        dialogue_masks.append(out.attention_mask)
                        dialogue_seg.append(torch.ones(out.input_ids.shape[1], dtype=int))

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

                        img_name = 'dia{}_utt{}_1.jpg'.format(row['Dialogue_ID'], row['Utterance_ID'])
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        dialogue_imgs.append(trans(img))

            return dialogue_tokens, dialogue_masks, dialogue_seg, dialogue_sen, dialogue_emo, dialogue_imgs

    elif dataset == 'MSED':
        if model_name == 'MMCNN' or model_name == 'BiGRU':
            dialogue_tokens = []
            dialogue_sen = []
            dialogue_emo = []
            dialogue_imgs = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['happiness', 'neutral', 'anger', 'disgust', 'fear', 'sad']
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for i, row in enumerate(csv_reader):                  
                        uttr = row['Caption']
                        dialogue_tokens.append(jieba.cut(uttr))

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

                        img_name = '{}.jpg'.format(i+1)
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        dialogue_imgs.append(trans(img))

            return dialogue_tokens, dialogue_sen, dialogue_emo, dialogue_imgs
        
        elif model_name == 'UPBMTL':
            dialogue_tokens = []
            dialogue_masks = []
            dialogue_sen = []
            dialogue_emo = []
            dialogue_imgs = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['happiness', 'neutral', 'anger', 'disgust', 'fear', 'sad']
            tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", cache_dir='/home/jiaao/data/albert-base-v2')
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for i, row in enumerate(csv_reader):
                        uttr = row['Caption']
                        out = tokenizer(uttr, add_special_tokens=True, max_length = 32, padding='max_length', truncation=True, return_tensors='np')
                        dialogue_tokens.append(out.input_ids)
                        dialogue_masks.append(out.attention_mask)

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

                        img_name = '{}.jpg'.format(i+1)
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        dialogue_imgs.append(trans(img))

            return dialogue_tokens, dialogue_masks, dialogue_sen, dialogue_emo, dialogue_imgs
        
        elif model_name == 'RoBERTa':
            dialogue_tokens = []
            dialogue_masks = []
            dialogue_sen = []
            dialogue_emo = []
            dialogue_imgs = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['happiness', 'neutral', 'anger', 'disgust', 'fear', 'sad']
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='/home/jiaao/data/roberta-base')
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for i, row in enumerate(csv_reader):
                        uttr = row['Caption']
                        out = tokenizer(uttr, add_special_tokens=True, max_length = 32, padding='max_length', truncation=True, return_tensors='np')
                        dialogue_tokens.append(out.input_ids)
                        dialogue_masks.append(out.attention_mask)

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

                        img_name = '{}.jpg'.format(i+1)
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        dialogue_imgs.append(trans(img))

            return dialogue_tokens, dialogue_masks, dialogue_sen, dialogue_emo, dialogue_imgs
        
        elif model_name == 'EfficientNet':
            dialogue_tokens = []
            dialogue_masks = []
            dialogue_sen = []
            dialogue_emo = []
            dialogue_imgs = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['happiness', 'neutral', 'anger', 'disgust', 'fear', 'sad']
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/home/jiaao/data/bert-base-uncased', do_lower_case=True)
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for i, row in enumerate(csv_reader):
                        uttr = row['Caption']
                        out = tokenizer(uttr, add_special_tokens=True, max_length = 32, padding='max_length', truncation=True, return_tensors='np')
                        dialogue_tokens.append(out.input_ids)
                        dialogue_masks.append(out.attention_mask)

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

                        img_name = '{}.jpg'.format(i+1)
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        dialogue_imgs.append(trans(img))

            return dialogue_tokens, dialogue_masks, dialogue_sen, dialogue_emo, dialogue_imgs
        
        elif model_name == 'mmbt':
            dialogue_tokens = []
            dialogue_masks = []
            dialogue_seg = []
            dialogue_sen = []
            dialogue_emo = []
            dialogue_imgs = []
            sen_list = ['positive', 'neutral', 'negative']
            emo_list = ['happiness', 'neutral', 'anger', 'disgust', 'fear', 'sad']
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/home/jiaao/data/bert-base-uncased', do_lower_case=True)
            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    fieldnames = next(reader)
                    csv_reader = csv.DictReader(f, fieldnames=fieldnames)

                    for i, row in enumerate(csv_reader):
                        uttr = row['Caption']
                        out = tokenizer(uttr, add_special_tokens=False, max_length = 32 - 2, padding='max_length', truncation=True, return_tensors='np')
                        dialogue_tokens.append(out.input_ids)
                        dialogue_masks.append(out.attention_mask)
                        dialogue_seg.append(torch.ones(out.input_ids.shape[1], dtype=int))

                        sen = row['Sentiment']
                        dialogue_sen.append(sen_list.index(sen))

                        emo = row['Emotion']
                        dialogue_emo.append(emo_list.index(emo))

                        img_name = '{}.jpg'.format(i+1)
                        img_path = os.path.join(img_dir, img_name)
                        assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                        img = Image.open(img_path)
                        dialogue_imgs.append(trans(img))

            return dialogue_tokens, dialogue_masks, dialogue_seg, dialogue_sen, dialogue_emo, dialogue_imgs

    dialogue_tokens = []
    dialogue_sen = []
    dialogue_emo = []
    dialogue_imgs = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/home/jiaao/data/bert-base-uncased')
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    sen_list = ['positive', 'neutral', 'negative']
    emo_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
            csv_reader = csv.DictReader(f, fieldnames=fieldnames)
            dialogue_id = 0
            uttr_tokens = []
            uttr_sen = []
            uttr_emo = []
            uttr_imgs = []

            for row in csv_reader:
                if row['Dialogue_ID'] != str(dialogue_id):
                    dialogue_tokens.append(uttr_tokens)
                    dialogue_sen.append(uttr_sen)
                    dialogue_emo.append(uttr_emo)
                    dialogue_imgs.append(uttr_imgs)

                    uttr_tokens = []
                    uttr_sen = []
                    uttr_emo = []
                    uttr_imgs = []

                    dialogue_id += 1
                
                uttr = row['Utterance']
                doc = nlp(uttr)
                tokens_ids = tokenizer.convert_tokens_to_ids([token.text.lower() for token in doc])

                tokens_ids.insert(0, 101)
                tokens_ids.append(102)
                uttr_tokens.append(tokens_ids)

                sen = row['Sentiment']
                uttr_sen.append(sen_list.index(sen))

                emo = row['Emotion']
                uttr_emo.append(emo_list.index(emo))

                img_name = 'dia{}_utt{}_1.jpg'.format(row['Dialogue_ID'],row['Utterance_ID'])
                img_path = os.path.join(img_dir, img_name)
                assert os.path.exists(img_path), 'Path: {} doesn\'t exist.'.format(img_path)
                img = Image.open(img_path)
                uttr_imgs.append(trans(img))
            
            dialogue_tokens.append(uttr_tokens)
            dialogue_sen.append(uttr_sen)
            dialogue_emo.append(uttr_emo)
            dialogue_imgs.append(torch.concatenate(uttr_imgs, dim=0))

    return dialogue_tokens, dialogue_sen, dialogue_emo, dialogue_imgs


def expand_list(n_list):
    e_list = []
    for element in n_list:
        if isinstance(element, (list, tuple)):
            e_list.extend(expand_list(element))
        else:
            e_list.append(element)
    return e_list


def process_emb(embedding, emb_dim):
    embeddings = {}
    embeddings["<PAD>"] = [0 for _ in range(emb_dim)]
    embeddings["<UNK>"] = np.random.uniform(-0.01,0.01,size = emb_dim).tolist()

    for emb in embedding:
        line = emb.strip().split()
        word = line[0]
        word_emb = [float(_) for _ in line[1:]]
        embeddings[word] = word_emb

    return  embeddings


def get_glove_emb(glove_path):
    embedding_file = open(glove_path, 'r', encoding='utf-8')
    embeddings = [emb.strip() for emb in embedding_file]
    embedding_file.close()

    emb_dim = int(glove_path.split('.')[-2].split('d')[0])

    embedding = process_emb(embeddings, emb_dim)

    return embedding


if __name__ == '__main__':
    # dialogue_adjs, dialogue_tokens, dialogue_sen, dialogue_emo = load('MELD', 'dev')
    # # print(dialogue_adjs[0], dialogue_tokens[0], dialogue_sen[0], dialogue_emo[0])
    # max_len = 0
    # for uttr_tokens in dialogue_tokens:
    #     for tokens in uttr_tokens:
    #         if len(tokens) > max_len:
    #             max_len = len(tokens)
    # print('max len is', max_len)
    job, sex, personality = get_speaker_info()
    print(job)
    print(sex)
    print(personality)