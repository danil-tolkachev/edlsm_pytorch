from __future__ import print_function
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
import cv2 as cv


class dataLoader(object):
    def __init__(self,
                 data_directory,
                 psz,
                 half_range,
                 nchannels=3,
                 mode='Train',
                 train_split_size=160):
        self.psz  = psz
        self.mode = mode
        self.half_range    = half_range
        self.data_directory = data_directory
        self.nchannels = nchannels
        self.train_split_size = train_split_size
        self.shift = 2
        self.get_data()

    def get_data(self):
        total_data = os.listdir(self.data_directory + '/disp_noc_0')
        indices = list(range(len(total_data)))
        random.seed(a=123)
        random.shuffle(indices)
        if self.mode == 'Train':
            self.img_set = indices[:self.train_split_size]
        else:
            self.img_set = indices[self.train_split_size:]

        points = []
        all_rgb_images_l = {}
        all_rgb_images_r = {}

        print('Preprocessing...')
        for img in tqdm(self.img_set):
            images_l_path = os.path.join(self.data_directory,
                                         'image_2/%06d_10.png' % img)
            images_r_path = os.path.join(self.data_directory,
                                         'image_3/%06d_10.png' % img)

            # read images into tensor
            image_l = Image.open(images_l_path)
            image_l = 255 * transforms.ToTensor()(image_l)
            image_r = Image.open(images_r_path)
            image_r = 255 * transforms.ToTensor()(image_r)

            image_l = (image_l - image_l.mean()) / image_l.std()
            image_r = (image_r - image_r.mean()) / image_r.std()

            all_rgb_images_l[img] = image_l
            all_rgb_images_r[img] = image_r
            if self.mode != 'Train':
                continue

            disp_path = os.path.join(self.data_directory,
                                     'disp_noc_0/%06d_10.png' % img)
            disp = cv.imread(disp_path, cv.IMREAD_UNCHANGED) / 256.0
            height, width = disp.shape
            psz = self.psz
            for v, u in zip(*np.where(disp > 0)):
                d = disp[v, u]
                ur = np.round(u - d)
                if (psz + self.shift < u < width - psz - self.shift
                        and psz + self.shift < ur < width - psz - self.shift
                        and psz <= v < height - psz):
                    points.append((img, u, v, d))

        self.points = points

        if self.mode == 'Test':
            self.max_steps = len(points)

        # Set to std deviation of 1
        self.all_rgb_images_l = all_rgb_images_l
        self.all_rgb_images_r = all_rgb_images_r

    def gen_random_data(self):
        self.epoch = 0
        while True:
            indices = list(range(len(self.points)))
            random.shuffle(indices)
            for i in indices:
                img_num, u, v, d = self.points[i]
                yield img_num, u, v, d
            self.epoch += 1


    def gen_data_batch(self, batch_size):
        data_gen = self.gen_random_data()

        while True:
            psz = self.psz
            psz2 = 2 * psz + 1
            half = self.half_range
            half2 = 2 * half

            image_l_batch = torch.zeros((batch_size, self.nchannels, psz2, psz2),
                                        dtype=torch.float)
            image_r_batch = torch.zeros(
                (batch_size, self.nchannels, psz2, psz2 + half2),
                dtype=torch.float)
            t_batch = torch.zeros((batch_size, 1), dtype=torch.int32)

            # Generate training batch
            for batch in range(batch_size):
                tr_num, u, v, d = next(data_gen)

                if random.random() <= 0.5:
                    l_image = self.all_rgb_images_l[tr_num]
                    r_image = self.all_rgb_images_r[tr_num]
                else: # swap and mirror
                    l_image = self.all_rgb_images_r[tr_num].flip(dims=(2,))
                    r_image = self.all_rgb_images_l[tr_num].flip(dims=(2,))
                    w = l_image.size(2)
                    u = round(w - u + d)

                v1, v2 = v - psz, v + psz + 1
                u1, u2 = u - psz, u + psz + 1
                u1r = max(0, u1 - half2 + self.shift)
                u2r = u1r + half2 + psz2
                t = round(u1 - d - u1r)

                image_l_batch[batch, ...] = l_image[:, v1:v2, u1:u2]
                image_r_batch[batch, ...] = r_image[:, v1:v2, u1r:u2r]
                t_batch[batch, 0] = t

            yield image_l_batch, image_r_batch, t_batch
