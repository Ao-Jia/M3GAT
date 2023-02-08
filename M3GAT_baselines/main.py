import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from models.MMCNN import MMCNN
from models.BiGRU import BiGRU
from models.UPBMTL import UPBMTL
from models.RoBERTa import RoBERTa
from models.EfficientNet import EfficientNet
from models.mmbt import MultimodalBertClf
import torch
from utils.data import dataset
from torch.utils.data import DataLoader
from transformers import AdamW
import time
from sklearn import metrics
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MSED', help='Dataset to be used.') # 'MELD'
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-5, help='Learning rate.')
parser.add_argument("--max_grad", type=float, default=10.0)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--nlayers_TE', type=int, default=1, help='Number of GAT layers in TextEncoder.')
parser.add_argument('--nlayers_IE', type=int, default=1, help='Number of GAT layers in ImageEncoder.')
parser.add_argument('--nlayers_MTI', type=int, default=1, help='Number of GAT layers in Speaker-Aware Multi-Task Interactive Conversation Graph.')
parser.add_argument('--z', type=int, default=1, help='Window size for connection in Speaker-Aware Multi-Task Interactive Conversation Graph.')
parser.add_argument('--sen_class', type=int, default=3)
parser.add_argument('--emo_class', type=int, default=6) # 7
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--scheduler_factor', type=float, default=0.1)
parser.add_argument('--scheduler_patience', type=int, default=3)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--sentence_max_size', type=int, default=32)
parser.add_argument('--glove_path', type=str, default='/home/jiaao/data/glove.6B/glove.6B.100d.txt')
parser.add_argument('--model_name', type=str, default='RoBERTa') # MMCNN, BiGRU, UPBMTL, RoBERTa, EfficientNet, mmbt
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _collate_func(instance_list):
    """
    As a function parameter to instantiate the DataLoader object.
    """
    n_entity = len(instance_list[0])
    scatter_b = [[] for _ in range(0, n_entity)]

    for idx in range(0, len(instance_list)):
        for jdx in range(0, n_entity):
            scatter_b[jdx].append(instance_list[idx][jdx])
    return scatter_b


def padding(tokens, masks):
    batch_size = len(tokens)
    len_list = []
    max_len = 0
    for i in range(batch_size):
        length = tokens[i].shape[0]
        len_list.append(length)
        if length > max_len:
            max_len = length

    for i in range(batch_size):
        length = len_list[i]
        if length == max_len:
            tokens[i] = tokens[i].reshape(1, max_len, -1)
            masks[i] = masks[i].reshape(1, max_len, -1)
            continue
        tokens[i] = np.concatenate((tokens[i], np.zeros((max_len-length, tokens[i].shape[1]), dtype='int')), axis=0).reshape(1, max_len, -1)
        masks[i] = np.concatenate((masks[i], np.zeros((max_len-length, masks[i].shape[1]), dtype='int')), axis=0).reshape(1, max_len, -1)

    return np.vstack(tokens), np.vstack(masks)


def evaluate(model_name, model, data_loader, test=False):
    model.eval()
    sen_loss_total = 0
    sen_predict_all = np.array([], dtype=int)
    sen_labels_all = np.array([], dtype=int)
    emo_loss_total = 0
    emo_predict_all = np.array([], dtype=int)
    emo_labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_loader:
            if model_name == 'MMCNN' or model_name == 'BiGRU':
                texts, sen_labels, emo_labels, imgs = batch

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    imgs = imgs.cuda()

                sen, emo = model.forward(texts, imgs)

            elif model_name == 'UPBMTL':
                texts, masks, sen_labels, emo_labels, imgs = batch

                texts = torch.from_numpy(np.vstack(texts))
                masks = torch.from_numpy(np.vstack(masks))
                sen_labels = torch.LongTensor(sen_labels)
                emo_labels = torch.LongTensor(emo_labels)
                imgs = torch.cat([img.unsqueeze(0) for img in imgs], dim=0)

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    imgs = imgs.cuda()
                
                sen, emo = model.forward(texts, masks, imgs)
            
            elif model_name == 'RoBERTa':
                texts, masks, sen_labels, emo_labels, imgs = batch

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    imgs = imgs.cuda()
                
                sen, emo = model.forward(texts, masks, imgs)
            
            elif model_name == 'EfficientNet':
                texts, masks, sen_labels, emo_labels, imgs = batch

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    imgs = imgs.cuda()

                sen, emo = model.forward(texts, masks, imgs)
            
            elif args.model_name == 'mmbt':
                texts, masks, segs, sen_labels, emo_labels, imgs = batch

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    segs = segs.cuda()
                    imgs = imgs.cuda()

                sen, emo = model.forward(texts, masks, segs, imgs)

            else:
                pass

            if torch.cuda.is_available():
                sen_labels = sen_labels.cuda()
                emo_labels = emo_labels.cuda()

            sen_loss = F.cross_entropy(sen, sen_labels)
            sen_labels = sen_labels.data.cpu().numpy()
            sen_loss_total += sen_loss
            
            emo_loss = F.cross_entropy(emo, emo_labels)
            emo_labels = emo_labels.data.cpu().numpy()
            emo_loss_total += emo_loss

            sen_predic = torch.max(sen.data, 1)[1].cpu().numpy()
            emo_predic = torch.max(emo.data, 1)[1].cpu().numpy()

            sen_labels_all = np.append(sen_labels_all, sen_labels)
            sen_predict_all = np.append(sen_predict_all, sen_predic)

            emo_labels_all = np.append(emo_labels_all, emo_labels)
            emo_predict_all = np.append(emo_predict_all, emo_predic)

    sen_acc = metrics.accuracy_score(sen_labels_all, sen_predict_all)
    emo_acc = metrics.accuracy_score(emo_labels_all, emo_predict_all)
    if test:
        if args.dataset == 'MELD':
            sentiment_list = ['positive', 'neutral', 'negative']
            emotion_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        elif args.dataset == 'MSED':
            sentiment_list = ['positive', 'neutral', 'negative']
            emotion_list = ['happiness', 'neutral', 'anger', 'disgust', 'fear', 'sad']
        sen_report = metrics.classification_report(sen_labels_all, sen_predict_all, labels=list(range(len(sentiment_list))), target_names=sentiment_list, digits=4)
        emo_report = metrics.classification_report(emo_labels_all, emo_predict_all, labels=list(range(len(emotion_list))), target_names=emotion_list, digits=4)
        
        sen_confusion = metrics.confusion_matrix(sen_labels_all, sen_predict_all)
        emo_confusion = metrics.confusion_matrix(emo_labels_all, emo_predict_all)
        return sen_acc, sen_loss_total / len(data_loader), sen_report, sen_confusion, emo_acc, emo_loss_total / len(data_loader), emo_report, emo_confusion
    return sen_acc, sen_loss_total / len(data_loader), emo_acc, emo_loss_total / len(data_loader)


def training(model_name, model, train_loader, dev_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        model.train()
        print("Training Epoch: {:4d}".format(epoch))
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            if args.model_name == 'MMCNN' or args.model_name == 'BiGRU':
                texts, sen_labels, emo_labels, imgs = batch

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    imgs = imgs.cuda()
                
                sen, emo = model.forward(texts, imgs)
            
            elif args.model_name == 'UPBMTL':
                texts, masks, sen_labels, emo_labels, imgs = batch

                texts = torch.from_numpy(np.vstack(texts))
                masks = torch.from_numpy(np.vstack(masks))
                sen_labels = torch.LongTensor(sen_labels)
                emo_labels = torch.LongTensor(emo_labels)
                imgs = torch.cat([img.unsqueeze(0) for img in imgs], dim=0)

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    imgs = imgs.cuda()
                
                sen, emo = model.forward(texts, masks, imgs)
            
            elif args.model_name == 'RoBERTa':
                texts, masks, sen_labels, emo_labels, imgs = batch

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    imgs = imgs.cuda()
                
                sen, emo = model.forward(texts, masks, imgs)
            
            elif args.model_name == 'EfficientNet':
                texts, masks, sen_labels, emo_labels, imgs = batch

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    imgs = imgs.cuda()

                sen, emo = model.forward(texts, masks, imgs)
            
            elif args.model_name == 'mmbt':
                texts, masks, segs, sen_labels, emo_labels, imgs = batch

                if torch.cuda.is_available():
                    texts = texts.cuda()
                    masks = masks.cuda()
                    segs = segs.cuda()
                    imgs = imgs.cuda()

                sen, emo = model.forward(texts, masks, segs, imgs)

            else:
                pass

            if torch.cuda.is_available():
                sen_labels = sen_labels.cuda()
                emo_labels = emo_labels.cuda()

            sen_loss = criterion(sen, sen_labels)
            emo_loss = criterion(emo, emo_labels)
            batch_loss = sen_loss + emo_loss
            total_loss += batch_loss.cpu().item()

            batch_loss.backward()
            optimizer.step()

        print('train_loss: {}'.format(total_loss))

        sen_dev_acc, sen_dev_loss, emo_dev_acc, emo_dev_loss = evaluate(model_name, model, dev_loader)
        print('\nsen_dev_acc: {},'.format(sen_dev_acc), 'sen_dev_loss: {}'.format(sen_dev_loss))
        print('\nemo_dev_acc: {},'.format(emo_dev_acc), 'emo_dev_loss: {}'.format(emo_dev_loss))
        
        scheduler.step(sen_dev_loss+emo_dev_loss)

    return model


if __name__ == '__main__':
    print(args)

    set_seed(args.seed)

    embedding = None
    train_set = dataset(args, embedding, 'train')
    dev_set = dataset(args, embedding, 'dev')
    test_set = dataset(args, embedding, 'test')

    if args.model_name == 'MMCNN' or args.model_name == 'BiGRU' or args.model_name == 'RoBERTa' or args.model_name == 'EfficientNet' or args.model_name == 'mmbt':
        train_loader = DataLoader(train_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        dev_loader = DataLoader(dev_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_func)
        dev_loader = DataLoader(dev_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_func)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_func)

    if args.model_name == 'MMCNN':
        model = MMCNN(args)
    elif args.model_name == 'BiGRU':
        model = BiGRU(args)
    elif args.model_name == 'UPBMTL':
        model = UPBMTL(args)
    elif args.model_name == 'RoBERTa':
        model = RoBERTa(args)
    elif args.model_name == 'EfficientNet':
        model = EfficientNet(args)
    elif args.model_name == 'mmbt':
        model = MultimodalBertClf(args)
    else:
        pass
    
    if torch.cuda.is_available():
        model = model.cuda()

    time_start = time.time()
    model = training(args.model_name, model, train_loader, dev_loader)
    print('train time is {}'.format(time.time() - time_start))

    sen_acc, sen_loss, sen_report, sen_confusion, emo_acc, emo_loss, emo_report, emo_confusion = evaluate(args.model_name, model, test_loader, test=True)
    sen_msg = 'Sentiment Test Loss: {0:>5.2},  Sentiment Test Acc: {1:>6.2%}'
    print(sen_msg.format(sen_loss, sen_acc))
    print("Precision, Recall and F1-Score...")
    print(sen_report)
    print("Confusion Matrix...")
    print(sen_confusion)

    emo_msg = 'Emotion Test Loss: {0:>5.2},  Emotion Test Acc: {1:>6.2%}'
    print(emo_msg.format(emo_loss, emo_acc))
    print("Precision, Recall and F1-Score...")
    print(emo_report)
    print("Confusion Matrix...")
    print(emo_confusion)