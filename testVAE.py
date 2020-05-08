import torch
import torchvision
from torch import nn, Tensor
from torch import optim
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


def reconstruct_img(templ_img, displ_field_fun):
    # templ_img = torch.Tensor(templ_img).cuda()
    #
    # displaced_img = torch.zeros((len(displ_field), 1, len(templ_img[0]), len(templ_img[1]))).cuda()
    #
    # for batch in range(0, len(displ_field_fun)):
    #
    #     tmp_p_x = displ_field_fun[batch, 0, :, :]
    #     tmp_p_y = displ_field_fun[batch, 1, :, :]
    #
    #     for i in range(0, len(templ_img[0])):
    #         for j in range(0, len(templ_img[1])):
    #             if (i + round(tmp_p_x[i, j].item()) < len(templ_img[0])) \
    #                     and round((j + tmp_p_y[i, j].item()) < len(templ_img[1])):
    #                 displaced_img[batch, 0, i, j] = templ_img[round(i + tmp_p_x[i, j].item()), round(j + tmp_p_y[i, j].item())]
    #             else:
    #                 displaced_img[batch, 0, i, j] = 0.0
    #
    # return displaced_img

    templ_img = torch.Tensor(templ_img).view(-1).cpu()
    new_idxs = torch.arange(0, 784).unsqueeze(0).unsqueeze(0).expand(len(displ_field_fun), 1, 784).cuda()
    displaced_img = torch.zeros((len(displ_field), 1, 784)).cpu()

    tmp_p_x = displ_field_fun[:, 0, :, :].view(-1, 1, 784).cuda()
    tmp_p_y = displ_field_fun[:, 1, :, :].view(-1, 1, 784).cuda() * 28

    new_idxs = new_idxs + tmp_p_x + tmp_p_y
    new_idxs = new_idxs.clamp(0, 783)
    new_idxs = new_idxs.cpu()

    for img in range(len(displ_field_fun)):
        for idx in range(0, 784):
            displaced_img[img, 0, round(new_idxs[img, 0, idx].item())] = templ_img[idx].item()

    mask_tens_0 = torch.zeros_like(displaced_img).cuda()
    mask_tens_1 = torch.ones_like(displaced_img).cuda()

    displaced_img = displaced_img.cuda()

    # displaced_img = displaced_img.where(displaced_img < 0, mask_tens_0)
    # displaced_img = displaced_img.where(displaced_img > 1, mask_tens_1)

    displaced_img = displaced_img.view(len(displ_field_fun), 1, 28, 28)

    return displaced_img

# variables:
num_epochs = 150

img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset2 = load_all_form_one_digit(8)
dataloader = DataLoader(dataset2, batch_size=128, shuffle=True)

dataset_templ = load_all_form_one_digit(4)

template = np.array(dataset_templ[0])


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