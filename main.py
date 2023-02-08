import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from model.models import M3GAT
import torch
from utils.data import dataset
from utils.load import expand_list, get_jsp_embedding, get_sp_jsp_list
from torch.utils.data import DataLoader
from transformers import AdamW
import time
from sklearn import metrics
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import random

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MELD', help='Dataset to be used.')
parser.add_argument('--task', type=str, default='s+e', help='\'s\' for sentiment, \'e\' for emotion, \'s+e\' for multi-task learning.')
parser.add_argument('--mode', type=str, default='t+v', help='\'t\' for text, \'v\' for video, \'t+v\' for multi-modal learning.')
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--bert_lr", type=float, default=4e-7, help='Learning rate for bert fine-tuning.')
parser.add_argument("--lr", type=float, default=3e-6, help='Learning rate.')
parser.add_argument("--max_grad", type=float, default=10.0)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--hidden_dim", type=int, default=768)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--nlayers_TE', type=int, default=1, help='Number of GAT layers in TextEncoder.')
parser.add_argument('--nlayers_IE', type=int, default=1, help='Number of GAT layers in ImageEncoder.')
parser.add_argument('--nlayers_MTI', type=int, default=1, help='Number of GAT layers in Speaker-Aware Multi-Task Interactive Conversation Graph.')
parser.add_argument('--z', type=int, default=3, help='Window size for connection in Speaker-Aware Multi-Task Interactive Conversation Graph.')
parser.add_argument('--sen_class', type=int, default=3)
parser.add_argument('--emo_class', type=int, default=7)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--scheduler_factor', type=float, default=0.1)
parser.add_argument('--scheduler_patience', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--seed', type=int, default=9)
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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


def evaluate(model, data_loader, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj, test=False):
    model.eval()
    sen_loss_total = 0
    sen_predict_all = np.array([], dtype=int)
    sen_labels_all = np.array([], dtype=int)
    emo_loss_total = 0
    emo_predict_all = np.array([], dtype=int)
    emo_labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for data_batch in data_loader:
            dialogue_adjs, dialogue_tokens, dialogue_sen, dialogue_emo, dialogue_imgs, dialogue_sp_idx = data_batch
            sen, emo = model.measure(dialogue_adjs, dialogue_tokens, dialogue_sen, dialogue_emo, dialogue_imgs, dialogue_sp_idx, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj, predic=True)
            dialogue_sen = torch.LongTensor(expand_list(dialogue_sen))
            dialogue_emo = torch.LongTensor(expand_list(dialogue_emo))
            if torch.cuda.is_available():
                dialogue_sen = dialogue_sen.cuda()
                dialogue_emo = dialogue_emo.cuda()

            if args.task == 's':
                sen_loss = F.cross_entropy(sen, dialogue_sen)
                sen_labels = dialogue_sen.data.cpu().numpy()
                sen_loss_total += sen_loss
                sen_predic = torch.max(sen.data, 1)[1].cpu().numpy()
                sen_labels_all = np.append(sen_labels_all, sen_labels)
                sen_predict_all = np.append(sen_predict_all, sen_predic)
            elif args.task == 'e':
                emo_loss = F.cross_entropy(emo, dialogue_emo)
                emo_labels = dialogue_emo.data.cpu().numpy()
                emo_loss_total += emo_loss
                emo_predic = torch.max(emo.data, 1)[1].cpu().numpy()
                emo_labels_all = np.append(emo_labels_all, emo_labels)
                emo_predict_all = np.append(emo_predict_all, emo_predic)
            else:
                sen_loss = F.cross_entropy(sen, dialogue_sen)
                sen_labels = dialogue_sen.data.cpu().numpy()
                sen_loss_total += sen_loss
                
                emo_loss = F.cross_entropy(emo, dialogue_emo)
                emo_labels = dialogue_emo.data.cpu().numpy()
                emo_loss_total += emo_loss

                sen_predic = torch.max(sen.data, 1)[1].cpu().numpy()
                emo_predic = torch.max(emo.data, 1)[1].cpu().numpy()

                sen_labels_all = np.append(sen_labels_all, sen_labels)
                sen_predict_all = np.append(sen_predict_all, sen_predic)

                emo_labels_all = np.append(emo_labels_all, emo_labels)
                emo_predict_all = np.append(emo_predict_all, emo_predic)

    if args.task == 's':
        sen_acc = metrics.accuracy_score(sen_labels_all, sen_predict_all)
        sen_f1 = metrics.f1_score(sen_labels_all, sen_predict_all, average="macro")
        emo_acc = None
        emo_f1 = None
    elif args.task == 'e':
        sen_acc = None
        sen_f1 = None
        emo_acc = metrics.accuracy_score(emo_labels_all, emo_predict_all)
        emo_f1 = metrics.f1_score(emo_labels_all, emo_predict_all, average="macro")
    else:
        sen_acc = metrics.accuracy_score(sen_labels_all, sen_predict_all)
        emo_acc = metrics.accuracy_score(emo_labels_all, emo_predict_all)
        sen_f1 = metrics.f1_score(sen_labels_all, sen_predict_all, average="macro")
        emo_f1 = metrics.f1_score(emo_labels_all, emo_predict_all, average="macro")

    if test:
        if args.dataset == 'MELD':
            sentiment_list = ['positive', 'neutral', 'negative']
            emotion_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        elif args.dataset == 'MSED':
            sentiment_list = ['positive', 'neutral', 'negative']
            emotion_list = ['happiness', 'neutral', 'anger', 'disgust', 'fear', 'sad']
        if args.task == 's':
            sen_report = metrics.classification_report(sen_labels_all, sen_predict_all, labels=list(range(len(sentiment_list))), target_names=sentiment_list, digits=4)
            sen_confusion = metrics.confusion_matrix(sen_labels_all, sen_predict_all)
            emo_report = None
            emo_confusion = None
        elif args.task == 'e':
            sen_report = None
            sen_confusion = None
            emo_report = metrics.classification_report(emo_labels_all, emo_predict_all, labels=list(range(len(emotion_list))), target_names=emotion_list, digits=4)
            emo_confusion = metrics.confusion_matrix(emo_labels_all, emo_predict_all)
        else:
            sen_report = metrics.classification_report(sen_labels_all, sen_predict_all, labels=list(range(len(sentiment_list))), target_names=sentiment_list, digits=4)
            emo_report = metrics.classification_report(emo_labels_all, emo_predict_all, labels=list(range(len(emotion_list))), target_names=emotion_list, digits=4)
            
            sen_confusion = metrics.confusion_matrix(sen_labels_all, sen_predict_all)
            emo_confusion = metrics.confusion_matrix(emo_labels_all, emo_predict_all)
        return sen_acc, sen_f1, sen_loss_total / len(data_loader), sen_report, sen_confusion, emo_acc, emo_f1, emo_loss_total / len(data_loader), emo_report, emo_confusion
    return sen_acc, sen_f1, sen_loss_total / len(data_loader), emo_acc, emo_f1, emo_loss_total / len(data_loader)


def training(model, train_loader, dev_loader, max_grad, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj):
    sen_best_dev_f1 = None
    sen_best_state = None

    emo_best_dev_f1 = None
    emo_best_state = None

    sen_best_epoch = -1
    emo_best_epoch = -1

    if args.dataset != 'MSED':
        bert_params_id = list(map(id, model.bert.parameters()))
        base_params = filter(lambda p: id(p) not in bert_params_id, model.parameters())
        optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': model.bert.parameters(), 'lr': args.bert_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    for epoch in range(args.epoch):
        model.train()
        print("Training Epoch: {:4d}".format(epoch))
        total_loss = 0.0

        for i, data_batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch_loss = model.measure(*data_batch, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj)
            total_loss += batch_loss.cpu().item()

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()

        print('train_loss: {}'.format(total_loss))

        sen_dev_acc, sen_dev_f1, sen_dev_loss, emo_dev_acc, emo_dev_f1, emo_dev_loss = evaluate(model, dev_loader, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj)
        if args.task != 'e':
            print('\nsen_dev_acc: {},'.format(sen_dev_acc), 'sen_dev_f1: {}'.format(sen_dev_f1), 'sen_dev_loss: {}'.format(sen_dev_loss))
            if sen_best_dev_f1 is None or sen_dev_f1 > sen_best_dev_f1:
                sen_best_dev_f1 = sen_dev_f1
                sen_best_state = copy.deepcopy(model.state_dict())
                sen_best_epoch = epoch
        if args.task != 's':
            print('\nemo_dev_acc: {},'.format(emo_dev_acc), 'emo_dev_f1: {}'.format(emo_dev_f1), 'emo_dev_loss: {}'.format(emo_dev_loss))
            if emo_best_dev_f1 is None or emo_dev_f1 > emo_best_dev_f1:
                emo_best_dev_f1 = emo_dev_f1
                emo_best_state = copy.deepcopy(model.state_dict())
                emo_best_epoch = epoch

        if args.task == 's':
            valid_loss = sen_dev_loss
        elif args.task == 'e':
            valid_loss = emo_dev_loss
        else:
            valid_loss = sen_dev_loss + emo_dev_loss

        scheduler.step(valid_loss)

    sen_ckpt = {
        "best_dev_f1": sen_best_dev_f1,
        "best_state": sen_best_state,
    }
    
    emo_ckpt = {
        "best_dev_f1": emo_best_dev_f1,
        "best_state": emo_best_state,
    }

    if args.task != 'e':
        print('sen_best_epoch: {}'.format(sen_best_epoch))
    if args.task != 's':
        print('emo_best_epoch: {}'.format(emo_best_epoch))

    return sen_ckpt, emo_ckpt, model

def gen_img_adj(h, w):
    adj = []
    for i in range(h*w):
        row = i // w
        col = i % w
        if (row - 1) >= 0:
            if (col - 1) >= 0:
                adj.append([i, i-w-1])
            adj.append([i, i-w])
            if (col + 1) < w:
                adj.append([i, i-w+1])
        if (col - 1) >= 0:
            adj.append([i, i-1])
        adj.append([i, i])
        if (col + 1) < w:
            adj.append([i, i+1])
        if (row + 1) < h:
            if (col - 1) >= 0:
                adj.append([i, i+w-1])
            adj.append([i, i+w])
            if (col + 1) < w:
                adj.append([i, i+w+1])
    adj = torch.LongTensor(adj)

    if torch.cuda.is_available():
        adj = adj.cuda()
        
    return adj


if __name__ == '__main__':
    print(args)

    set_seed(args.seed)

    train_set = dataset(args.dataset, 'train')
    train_loader = DataLoader(train_set, args.batch_size, shuffle=False, collate_fn=_collate_func, num_workers=args.num_workers, pin_memory=True)
    dev_set = dataset(args.dataset, 'dev')
    dev_loader = DataLoader(dev_set, args.batch_size, shuffle=False, collate_fn=_collate_func, num_workers=args.num_workers, pin_memory=True)
    test_set = dataset(args.dataset, 'test')
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, collate_fn=_collate_func, num_workers=args.num_workers, pin_memory=True)

    job_embedding, sex_embedding, personality_embedding = get_jsp_embedding()
    sp_jsp_list = get_sp_jsp_list()

    img_adj = gen_img_adj(180,320)

    model = M3GAT(dropout=args.dropout,
                  hidden_dim=args.hidden_dim,
                  alpha=args.alpha,
                  nheads=args.nheads,
                  nlayers_TE=args.nlayers_TE,
                  nlayers_IE=args.nlayers_IE,
                  nlayers_MTI=args.nlayers_MTI,
                  z=args.z,
                  sen_class=args.sen_class,
                  emo_class=args.emo_class,
                  dataset=args.dataset,
                  mode=args.mode,
                  task=args.task)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.dataset == 'MELD':
        sen_best_model_file = './save/sen_best.pt'
        emo_best_model_file = './save/emo_best.pt'
    elif args.dataset == 'MSED':
        sen_best_model_file = './save/sen_best_MSED.pt'
        emo_best_model_file = './save/emo_best_MSED.pt'

    time_start = time.time()
    sen_ckpt, emo_ckpt, model = training(model, train_loader, dev_loader, args.max_grad, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj)
    print('train time is {}'.format(time.time() - time_start))

    # last model testing
    sen_acc, sen_f1, sen_loss, sen_report, sen_confusion, emo_acc, emo_f1, emo_loss, emo_report, emo_confusion = evaluate(model, test_loader, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj, test=True)
    
    if args.task != 'e':
        sen_msg = 'Sentiment Test Loss: {0:>5.2},  Sentiment Test Acc: {1:>6.2%}'
        print(sen_msg.format(sen_loss, sen_acc))
        print("Precision, Recall and F1-Score...")
        print(sen_report)
        print("Confusion Matrix...")
        print(sen_confusion)

    if args.task != 's':
        emo_msg = 'Emotion Test Loss: {0:>5.2},  Emotion Test Acc: {1:>6.2%}'
        print(emo_msg.format(emo_loss, emo_acc))
        print("Precision, Recall and F1-Score...")
        print(emo_report)
        print("Confusion Matrix...")
        print(emo_confusion)

    # best sen model testing
    if args.task != 'e':
        model.load_state_dict(sen_ckpt['best_state'])
        sen_acc, sen_f1, sen_loss, sen_report, sen_confusion, emo_acc, emo_f1, emo_loss, emo_report, emo_confusion = evaluate(model, test_loader, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj, test=True)
        sen_msg = 'Sentiment Test Loss: {0:>5.2},  Sentiment Test Acc: {1:>6.2%}'
        print(sen_msg.format(sen_loss, sen_acc))
        print("Precision, Recall and F1-Score...")
        print(sen_report)
        print("Confusion Matrix...")
        print(sen_confusion)

        # saving
        if os.path.exists(sen_best_model_file):
            ckpt = torch.load(sen_best_model_file)
            if sen_f1 > ckpt['best_dev_f1']:
                sen_ckpt['best_dev_f1'] = sen_f1
                torch.save(sen_ckpt, sen_best_model_file)
        else:
            sen_ckpt['best_dev_f1'] = sen_f1
            torch.save(sen_ckpt, sen_best_model_file)

    # best emo model testing
    if args.task != 's':
        model.load_state_dict(emo_ckpt['best_state'])
        sen_acc, sen_f1, sen_loss, sen_report, sen_confusion, emo_acc, emo_f1, emo_loss, emo_report, emo_confusion = evaluate(model, test_loader, job_embedding, sex_embedding, personality_embedding, sp_jsp_list, img_adj, test=True)
        emo_msg = 'Emotion Test Loss: {0:>5.2},  Emotion Test Acc: {1:>6.2%}'
        print(emo_msg.format(emo_loss, emo_acc))
        print("Precision, Recall and F1-Score...")
        print(emo_report)
        print("Confusion Matrix...")
        print(emo_confusion)

        # saving
        if os.path.exists(emo_best_model_file):
            ckpt = torch.load(emo_best_model_file)
            if emo_f1 > ckpt['best_dev_f1']:
                emo_ckpt['best_dev_f1'] = emo_f1
                torch.save(emo_ckpt, emo_best_model_file)
        else:
            emo_ckpt['best_dev_f1'] = emo_f1
            torch.save(emo_ckpt, emo_best_model_file)