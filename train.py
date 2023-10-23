import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss
from dataloader import TrainDataset, ValidDataset
from torch.utils.data import DataLoader
from model import Unet
from torchsummary import summary
from tqdm.auto import tqdm, trange
from torchvision.utils import make_grid
import numpy as np
from misc import overlay, Metrics
import os
import time
device = torch.device('cuda')
model = Unet()
model = model.to(device)
summary(model, (3, 1080, 1920))

opt = Adam(model.parameters(), lr=1e-3)
loss_fn = MSELoss()

train = TrainDataset(2048)
trainloader = DataLoader(
    train,
    batch_size=256,#16, 18
    num_workers=4,
    drop_last=True,
    prefetch_factor=12,
)
valid = ValidDataset()
validloader = DataLoader(
    valid,
    batch_size = len(valid)
)

writer = SummaryWriter()

loss_fn = loss_fn.to(device)
lossLogger = Metrics(batch_size=256)
for epoch in range(250):
    for i, batch in enumerate(tqdm(trainloader)):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lossLogger.add(loss.item())

        if i == 0 and epoch > 0:
            grid = make_grid(x[:8]/255)
            writer.add_image('X', grid, epoch)
            grid = make_grid(y[:8])
            writer.add_image('Y', grid, epoch)
            grid = make_grid(outputs[:8])
            writer.add_image('Pred', grid, epoch)
            grid = overlay(x[:8], outputs[:8])
            writer.add_image('Overlay', grid, epoch)

    writer.add_scalar('Loss', lossLogger.get(), epoch)

    for i, batch in enumerate(validloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        grid = make_grid(x/255)
        writer.add_image('Val_X', grid, epoch)
        grid = make_grid(y)
        writer.add_image('Val_Y', grid, epoch)
        grid = make_grid(outputs)
        writer.add_image('Val_Pred', grid, epoch)
        grid = overlay(x, outputs)
        writer.add_image('Val_Overlay', grid, epoch)
        writer.add_scalar('Val_Loss', loss.item() / len(batch), epoch)