import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

class Metrics:
    def __init__(self, batch_size):
        self.batch_size = batch_size 
        self.loss = 0
        self.length = 0

    def add(self, loss):
        self.loss += loss / self.batch_size
        self.length += 1

    def get(self):
        output = self.loss / self.length
        self.length = 0
        self.loss = 0
        return output


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def overlay(x, outputs):
    x = make_grid(x)
    outputs = make_grid(outputs)
    x = x.cpu().numpy()
    outputs = outputs.cpu().numpy()

    outputs = outputs.swapaxes(0, 2)
    # outputs = outputs.swapaxes(0, 1)

    outputs = cv2.resize(outputs, (x.shape[1], x.shape[2]))

    outputs = outputs.swapaxes(0, 2)
    # outputs = outputs.swapaxes(1, 2)
    # exit(0)

    x /= 255
    x += outputs

    x = np.fmin(1, x)
    
    return x

def concat(inpath='Train/Seperate', outpath='Train/video.mp4'):
    videos = []
    for filename in os.listdir(inpath):
        path = os.path.join(inpath, filename)
        videos.append(VideoFileClip(path))
    output = concatenate_videoclips(videos)
    output = output.without_audio()
    output = output.set_fps(5)
    output.write_videofile(outpath)