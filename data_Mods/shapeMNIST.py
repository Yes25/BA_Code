import numpy as np

from torchvision import transforms
from torchvision.datasets import MNIST


def load_all_form_one_digit(digit=8):
    dataset = MNIST(root='../../data', train = True, download=True)

    list_img_label = []

    for e in dataset:
        if e[1] == digit:
            list_img_label.append(e[0])

    return list_img_label


def build_mean_shape(img_list):

    mean_shape_img = np.zeros((28,28))

    for e in img_list:
        mean_shape_img = mean_shape_img + np.array(e)

    mean_shape_img = mean_shape_img / len(img_list)

    # Um die Kanten dünner zu machen? sieht aber scheiße aus... Wenn eher erosion oder so...
    # threshold_arr = mean_shape_img[:, :] > 105
    # mean_shape_img = mean_shape_img * threshold_arr

    return mean_shape_img








