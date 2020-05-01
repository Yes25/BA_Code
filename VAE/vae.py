import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        self.enc = Enoder()
        self.dec = Decoder()

    def forward(self, x):
        # TODO
        pass


class Enoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(1,4,kernel_size=2,stride=2,padding=0,bias=True)
        self.conv2 = nn.Conv2d(4,8,kernel_size=2,stride = 2,padding=0,bias= True)
        self.conv3 = nn.Conv2d(8,16,kernel_size=2,stride=2,padding=0,bias=True)
        self.conv4 = nn.Conv2d(16,16,kernel_size=2,stride=2,padding=0,bias=True)
        # TODO: größe des latent space ausrechnen; abhängig von einem n?
        self.linear = nn.Linear(16, latent_dim)

    def forward(self, x):

        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        x = self.linear(x)

        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        # TODO wie oben... Größe latent space
        self.linear = nn.Linear(latent_dim, 16)
        self.de_conv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0, bias=True)
        self.de_conv2 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, padding=0, bias=True)
        self.de_conv3 = nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, x):

        x = F.tanh(self.linear(x))
        x = F.tanh(self.de_conv1(x))
        x = F.tanh(self.de_conv2(x))
        x = self.de_conv3(x)

        return x
