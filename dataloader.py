import torch
import os
import cv2
import warnings
import numpy as np
from torch.utils.data import Dataset
import pims
from misc import softmax
import time
import random
class TrainDataset(Dataset):
    def __init__(self, length, PATH='Train/video.mp4'):
        # print("loading dataset...")
        self.length = length
        # self.imgs = pims.Video(PATH)
        self.imgs = []
        cap = cv2.VideoCapture(PATH)
        while True:
            read, frame = cap.read()
            if not read:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame)
            frame = frame.astype(np.float32)
            self.imgs.append(frame)

        self.length_video_length = len(self.imgs) - 5

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        index = np.random.randint(0, self.length_video_length)
        img = self.imgs[index]
        # try:
        #     index = np.random.randint(0, self.length_video_length)
        #     img = self.imgs[index]
        # except:
        #     index = np.random.randint(0, self.length_video_length)
        #     img = self.imgs[index]

        (corners, ids, rejected) = cv2.aruco.detectMarkers(img, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000))
        label = np.zeros(shape=(img.shape[0], img.shape[1]))
        if ids == [0]:
            corners = corners[0].astype(np.int32)
            label = cv2.fillPoly(label, pts = corners, color =(255,255,255))
        label = cv2.resize(label, (671, 351))

        img = img.swapaxes(0, 2)
        img = img.swapaxes(1, 2)
        
        label = softmax(label)
        label *= 23552 #671 * 351 / 10
        label = np.array([label])

        return img, label

class ValidDataset(Dataset):
    def __init__(self, PATH='Valid'):
        self.x = []
        self.y = []
        for filename in os.listdir(PATH):
            img = cv2.imread(os.path.join(PATH, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img[:1080*4, :1920*4]
            img = cv2.resize(img, (1920, 1080))
            (corners, ids, rejected) = cv2.aruco.detectMarkers(img, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000))
            label = np.zeros(shape=(img.shape[0], img.shape[1]))
            if ids == [0]:
                corners = corners[0].astype(np.int32)
                label = cv2.fillPoly(label, pts = corners, color =(255,255,255))
            img = img.astype(np.float32)
            img = img.swapaxes(0, 2)
            img = img.swapaxes(1, 2)

            img = img.astype(np.float32)
            label = label.astype(np.float32)
            label = cv2.resize(label, (671, 351))
            label = softmax(label)
            label = np.array([label])

            self.x.append(img)
            self.y.append(label)

    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

        

    

