import torch
import torchvision
from torch import nn, Tensor
from torch import optim
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt

import os

from VAE.vae import *
from data_Mods.shapeMNIST import all_from_one_digit_as_tensor
from data_Mods.shapeMNIST import load_all_form_one_digit

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# variables:
num_epochs = 150

img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset2 = load_all_form_one_digit(8)
dataloader = DataLoader(dataset2, batch_size=128, shuffle=True)

dataset_templ = load_all_form_one_digit(0)

template_img = np.array(dataset_templ[0])


model = VAE(latent_dim=20, template=template_img)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(dataloader):
        img_batch = data.unsqueeze(dim=1)
        img_batch = img_batch.cuda()

        optimizer.zero_grad()
        displ_field_run, recon_batch,  mu_run, logvar_run = model(img_batch)

        loss = loss_function(recon_batch, img_batch, displ_field_run, mu_run, logvar_run)
        # loss = Variable(loss, requires_grad=True)
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
    if epoch % 2 == 0:
        save = to_img(recon_batch.cpu().data)
        save_image(save, './vae_img/image_{}.png'.format(epoch))
