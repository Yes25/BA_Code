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
from data_Mods.shapeMNIST import all_from_one_digit_as_tensor
from data_Mods.shapeMNIST import load_all_form_one_digit

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# variables:
num_epochs = 50

img_transform = transforms.Compose([
    transforms.ToTensor()
])

#dataset1 = MNIST(root='../../data', train=True, download=True, transform=img_transform)
#print(type(dataset1))
#dataset = all_from_one_digit_as_tensor(8)
#dataset = dataset.unsqueeze(dim=1)
#print(dataset.shape)
#dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

dataset2 = load_all_form_one_digit(8)
dataloader = DataLoader(dataset2, batch_size=128, shuffle=True)



model = VAE(latent_dim=3)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(dataloader):
        img_batch = data.unsqueeze(dim=1)
        img_batch = img_batch.to(device='cuda', dtype=torch.float)

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