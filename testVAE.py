import torch
import torchvision
from torch import nn
from torch import optim
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np

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

def reconstruct_img(templ_img, displ_field):

    displaced_img = np.ndarray(shape=(len(displ_field), len(templ_img[0]), len(templ_img[1])), dtype=float)

    for batch in range(0, len(displ_field)):

        tmp_p_x = displ_field[batch, 0, :, :]
        tmp_p_y = displ_field[batch, 1, :, :]

        for i in range(0, len(templ_img[0])):
            for j in range(0, len(templ_img[1])):
                if (i + round(tmp_p_x[i, j].item()) < len(templ_img[0])) \
                        and round((j + tmp_p_y[i, j].item()) < len(templ_img[1])):
                    displaced_img[batch, i, j] = templ_img[round(i + tmp_p_x[i, j].item()), round(j + tmp_p_y[i, j].item())]
                else:
                    displaced_img[batch, i, j] = 0.0

    displaced_img = torch.tensor(displaced_img, dtype=torch.float)
    displaced_img = displaced_img.unsqueeze(1)
    # print(displaced_img[63])
    return displaced_img

# variables:
num_epochs = 20

img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset2 = load_all_form_one_digit(8)
dataloader = DataLoader(dataset2, batch_size=64, shuffle=True)

template = np.array(dataset2[0])
print(template.shape)

model = VAE(latent_dim=3)
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
        displ_field, mu, logvar = model(img_batch)
        recon_batch = reconstruct_img(template, displ_field)
        recon_batch = recon_batch.cuda()
        img_batch = img_batch.cuda()
        loss = loss_function(recon_batch, img_batch, mu, logvar)
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(img_batch.shape)
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