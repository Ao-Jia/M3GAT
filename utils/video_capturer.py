import os
import cv2
import numpy as np

modes = ['train', 'dev', 'test']
for mode in modes:
    video_dir = './data/MELD/{}_video'.format(mode)
    out_dir = './data/MELD/{}_img'.format(mode)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    assert os.path.exists(video_dir)
    for video_name in os.listdir(video_dir):
        if mode == 'train' and video_name == 'dia125_utt3.mp4':
            frame = np.zeros((720,1280,3),dtype='int')
            filename = video_name.split('.')[0] + '_{}.jpg'.format(0)
            cv2.imwrite(os.path.join(out_dir, filename), frame)
            filename = video_name.split('.')[0] + '_{}.jpg'.format(1)
            cv2.imwrite(os.path.join(out_dir, filename), frame)
            filename = video_name.split('.')[0] + '_{}.jpg'.format(2)
            cv2.imwrite(os.path.join(out_dir, filename), frame)
            continue
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), 'Path: {} doesn\'t exist.'.format(video_path)
        cnt = 0
        num = 0
        frames_num=cap.get(7)
        if frames_num == 2:
            ret, frame = cap.read()
            filename = video_name.split('.')[0] + '_{}.jpg'.format(0)
            if not os.path.exists(os.path.join(out_dir, filename)):
                print('dealing {}...'.format(filename))
                cv2.imwrite(os.path.join(out_dir, filename), frame)
            ret, frame = cap.read()
            filename = video_name.split('.')[0] + '_{}.jpg'.format(1)
            if not os.path.exists(os.path.join(out_dir, filename)):
                print('dealing {}...'.format(filename))
                cv2.imwrite(os.path.join(out_dir, filename), frame)
            filename = video_name.split('.')[0] + '_{}.jpg'.format(2)
            if not os.path.exists(os.path.join(out_dir, filename)):
                print('dealing {}...'.format(filename))
                cv2.imwrite(os.path.join(out_dir, filename), frame)
            num = 3
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frames_num <= 4:
                start = 0
            elif frames_num <= 11:
                start = 1
            else:
                start = 4
            if cnt == start or cnt == frames_num // 2 or cnt == frames_num - start - 1:
                filename = video_name.split('.')[0] + '_{}.jpg'.format(num)
                num += 1
                if not os.path.exists(os.path.join(out_dir, filename)):
                    print('dealing {}...'.format(filename))
                    cv2.imwrite(os.path.join(out_dir, filename), frame)
            cnt += 1
        assert num == 3, 'the frames_num of {} is {}.'.format(video_name, frames_num)

print('success.')