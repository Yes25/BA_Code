import numpy as np
import random


class displField():

    def __init__(self, x, y):

        self.disp_field = np.ndarray(shape=(x,y), dtype=tuple)
        self.disp_field.fill((0,0))

    def gen_rand_displ_field(self):
        self.disp_field.fill((random.randint(-27,27),random.randint(-27,27)))

    def displace(self, img):
        displaced_img = np.ndarray(shape=(len(img[0]),len(img[1])), dtype=int)

        for i in range(0, len(img[0])):
            for j in range(0, len(img[1])):
                if i + self.disp_field[i,j][0] < len(img[0]) and j + self.disp_field[i,j][1] < len(img[1]):
                    displaced_img[i,j] = img[i + self.disp_field[i,j][0], j + self.disp_field[i,j][1]]
                else:
                    displaced_img[i, j] = 0

        return displaced_img
