import torch
import torchvision
from torch import nn
from torch import optim
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import os

from VAE.vae import *

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# variables:
num_epochs = 100

img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = MNIST(root='../../data', train=True, download=True, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = VAE(latent_dim=20)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(dataloader):
        img_batch = data[0]
        img_batch = img_batch.cuda()

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img_batch)
        loss = loss_function(recon_batch, img_batch, mu, logvar)
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(img_batch),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                loss.data.item() / len(img_batch)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset)))
    if epoch % 10 == 0:
        save = to_img(recon_batch.cpu().data)
        save_image(save, './vae_img/image_{}.png'.format(epoch))