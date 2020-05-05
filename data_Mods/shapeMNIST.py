import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST


def load_all_form_one_digit(digit=8):
    """returns a list of images all with the same given label"""
    dataset = MNIST(root='../../data', train = True, download=True)

    list_img_label = []

    for e in dataset:
        if e[1] == digit:
            list_img_label.append(np.array(e[0]))

    return list_img_label

def all_from_one_digit_as_tensor(digit=8):

    img_list = load_all_form_one_digit(digit)
    all_img_as_arr = np.stack(img_list, axis=0)
    all_img_as_tensor = torch.tensor(all_img_as_arr)
    return all_img_as_tensor


def build_mean_shape(img_list):

    mean_shape_img = np.zeros((28,28))

    for e in img_list:
        mean_shape_img = mean_shape_img + np.array(e)

    mean_shape_img = mean_shape_img / len(img_list)

    # Um die Kanten dünner zu machen? sieht aber scheiße aus... Wenn eher erosion oder so...
    # threshold_arr = mean_shape_img[:, :] > 105
    # mean_shape_img = mean_shape_img * threshold_arr

    return mean_shape_img








