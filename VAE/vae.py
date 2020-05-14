import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

############## Variables ##############
kern_sz = 3 # 2
stride = 1 # 2
smallest_img_dim = 22 # 3

def reconstruct_img(templ_img, displ_field_fun):

    # templ_img = torch.Tensor(templ_img).view(-1).cpu()
    # new_idxs = torch.arange(0, 784).unsqueeze(0).unsqueeze(0).expand(len(displ_field_fun), 1, 784).cuda()
    # displaced_img = torch.zeros(len(displ_field_fun), 1, 784).cpu()
    #
    # tmp_p_x = displ_field_fun[:, 0, :, :].view(-1, 1, 784).cuda()
    # tmp_p_y = displ_field_fun[:, 1, :, :].view(-1, 1, 784).cuda() * 28
    #
    # new_idxs = new_idxs + tmp_p_x + tmp_p_y
    # new_idxs = new_idxs.clamp(0, 783)
    # new_idxs = new_idxs.cpu()
    #
    # for img in range(len(displ_field_fun)):
    #     for idx in range(0, 784):
    #         displaced_img[img, 0, round(new_idxs[img, 0, idx].item())] = templ_img[idx].item()
    #
    # displaced_img = displaced_img.cuda()
    #
    # displaced_img = displaced_img.view(len(displ_field_fun), 1, 28, 28)
    #
    # return Variable(displaced_img, requires_grad=True)

    num_imges = len(displ_field_fun)

    template_img_batch = torch.Tensor(templ_img).expand(num_imges,28,28).cuda()
    template_img_batch = template_img_batch.unsqueeze(1)

    rec_img_batch = F.grid_sample(template_img_batch, displ_field_fun,mode='bilinear' ,  padding_mode='zeros')

    rec_img_batch = rec_img_batch.view(-1, 1, 28, 28)

    return rec_img_batch

def loss_function(recon_img, input_img, disp_field, mu, logvar):
    rec_func = nn.MSELoss(reduction='sum')

    in_out_diff = rec_func(recon_img, input_img)

    kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kld = torch.sum(kld_element).mul_(-0.5)

    ### Loss of the displacmentfield ###
    x_vals = (disp_field[:, 0, :, :] ** 2).cuda()
    y_vals = (disp_field[:, 1, :, :] ** 2).cuda()

    length_vs = ((x_vals + y_vals) ** 0.5).cuda()

    mask_tens = torch.zeros_like(length_vs).cuda()
    reg_field_loss = (rec_func(length_vs, mask_tens)**0.5).cuda()

    return in_out_diff + kld + reg_field_loss


class VAE(nn.Module):

    def __init__(self, latent_dim, template):
        super(VAE, self).__init__()

        self.enc = Enoder(latent_dim)
        self.dec = Decoder(latent_dim, template)

    def reparam(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        z = eps * std + mu
        return z

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        displ_field, recon_img = self.dec(z)
        return displ_field, recon_img, mu, logvar


class Enoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=kern_sz, stride=stride, padding=0, bias=True)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=kern_sz, stride=stride, padding=0, bias=True)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=kern_sz, stride=stride, padding=0, bias=True)
        # self.conv4 = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0, bias=True)
        self.linear_1 = nn.Linear(smallest_img_dim * smallest_img_dim * 16, latent_dim)
        self.linear_2 = nn.Linear(smallest_img_dim * smallest_img_dim * 16, latent_dim)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        # x = F.tanh(self.conv4(x))
        z_mu = self.linear_1(x.view(x.size(0), -1))
        z_var = self.linear_2(x.view(x.size(0), -1))

        return z_mu, z_var


class Decoder(nn.Module):

    def __init__(self, latent_dim, template):
        super().__init__()

        self.template = template

        self.linear = nn.Linear(latent_dim, smallest_img_dim * smallest_img_dim * 16)
        self.de_conv1 = nn.ConvTranspose2d(16, 8, kernel_size=kern_sz, stride=stride, padding=0, bias=True)
        self.de_conv2 = nn.ConvTranspose2d(8, 4, kernel_size=kern_sz, stride=stride, padding=0, bias=True)
        self.de_conv3 = nn.ConvTranspose2d(4, 2, kernel_size=kern_sz, stride=stride, padding=0, bias=True)

    def forward(self, x):
        x = torch.tanh(self.linear(x))
        x = torch.tanh(self.de_conv1(x.view(-1, 16, smallest_img_dim, smallest_img_dim)))
        x = torch.tanh(self.de_conv2(x))
        ret_displ_field = torch.tanh(self.de_conv3(x))
        ret_displ_field = ret_displ_field.view(-1, 28, 28, 2)
        recon_img = reconstruct_img(templ_img=self.template, displ_field_fun=ret_displ_field)

        return ret_displ_field, recon_img


