# M3GAT

Data preparation
1. Decompress the dataset in./data folder
2. As for MELD dataset, use python ./utils/video_capturer.py to preprocess the video data

Training
MELD: python main.py --dataset MELD --epoch 50 --batch_size 2 --bert_lr 4e-7 --lr 3e-6 --sen_class 3 --emo_class 7
MEISD: python main.py --dataset MEISD --epoch 100 --batch_size 2 --bert_lr 4e-7 --lr 3e-6 --sen_class 3 --emo_class 9
MSED: python main.py --dataset MSED --epoch 50 --batch_size 16 --bert_lr 1e-5 --lr 1e-5 --sen_class 3 --emo_class 6
