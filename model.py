import torch
import torch.nn as nn

class Unet(torch.nn.Module):

    def __init__(self, ch=16):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv2d(3, ch, (3, 3))
        self.conv2 = nn.Conv2d(ch, ch, (3, 3))
        self.conv3 = nn.Conv2d(ch, ch, (3, 3))
        self.conv4 = nn.Conv2d(ch, ch, (3, 3))
        self.conv5 = nn.Conv2d(ch, ch, (3, 3))

        self.up1 = nn.ConvTranspose2d(ch, ch, (3,3), 2)
        self.up2 = nn.ConvTranspose2d(ch, ch, (3,3), 2)
        self.up3 = nn.ConvTranspose2d(ch, ch, (3,3), 2)
        self.up4 = nn.ConvTranspose2d(ch, ch, (3,3), 2)
        self.up5 = nn.ConvTranspose2d(ch, 1, (3,3), 2)


        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(3, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.activation(x)

        #--------------------------------------------

        x = self.up1(x)
        x = self.activation(x)

        x = self.up2(x)
        x = self.activation(x)

        x = self.up3(x)
        x = self.activation(x)

        x = self.up4(x)
        x = self.activation(x)

        x = self.up5(x)
        x = self.sigmoid(x)

        return x
