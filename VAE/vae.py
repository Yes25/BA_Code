import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F



def loss_function(recon_img, input_img, mu, logvar):
    rec_func = nn.MSELoss(reduction='sum')
    bce = F.binary_cross_entropy(recon_img, input_img);
    kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kld = torch.sum(kld_element).mul_(-0.5)
    return bce + kld


class VAE(nn.Module):

    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.enc = Enoder(latent_dim)
        self.dec = Decoder(latent_dim)

    def reparam(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std).cuda()
        z = eps * std + mu
        return z

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        displ_field = self.dec(z)
        return displ_field, mu, logvar


class Enoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0, bias=True)
        # self.conv4 = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0, bias=True)
        self.linear_1 = nn.Linear(22 * 22 * 16, latent_dim)
        self.linear_2 = nn.Linear(22 * 22 * 16, latent_dim)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        # x = F.tanh(self.conv4(x))
        z_mu = self.linear_1(x.view(x.size(0), -1))
        z_var = self.linear_2(x.view(x.size(0), -1))

        return z_mu, z_var


class Decoder(nn.Module):

    def __init__(self, latent_dim,):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 22 * 22 * 16)
        self.de_conv1 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=0, bias=True)
        self.de_conv2 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=0, bias=True)
        self.de_conv3 = nn.ConvTranspose2d(4, 2, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = F.tanh(self.linear(x))
        x = F.tanh(self.de_conv1(x.view(-1, 16, 22, 22)))
        x = F.tanh(self.de_conv2(x))
        displ_field = self.de_conv3(x)

        return displ_field


