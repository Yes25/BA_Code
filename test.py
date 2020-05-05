import torch
import numpy as np
import matplotlib.pyplot as plt

from data_Mods.shapeMNIST import load_all_form_one_digit
from data_Mods.shapeMNIST import all_from_one_digit_as_tensor
from data_Mods.displfield import *

###### data shaping testcode ########

img_as_tens = all_from_one_digit_as_tensor(8)
print(img_as_tens.shape)




###### disp field tests ############
list_img = load_all_form_one_digit(8)
print(len(list_img))

x = displField(28, 28)
print(x.disp_field.shape)

x.gen_rand_displ_field()
print(x.disp_field[:,:])

print(x.disp_field[0,0][0])

displ_img = x.displace(np.array(list_img[0]))
plt.imshow(displ_img,'gray')
plt.show()

