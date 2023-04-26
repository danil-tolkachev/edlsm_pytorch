from __future__ import print_function
import os
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.nets_test import Inference
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",
                    type=str,
                    default='./data_scene_flow/training',
                    help="where the dataset is stored")
parser.add_argument("--save_root",
                    type=str,
                    default='./dataset',
                    help="Where to dump the data")
parser.add_argument("--checkpoint_dir",
                    type=str,
                    default='./saved_models/kitti_b128_3pxloss',
                    help="Where the ckpt files are")
parser.add_argument("--checkpoint_file",
                    type=str,
                    default='edlsm_38000.ckpt',
                    help="checkpoint file name to load")
parser.add_argument("--resize_image",
                    type=str,
                    default='True',
                    help="Resize image")
parser.add_argument("--test_num",
                    type=int,
                    default=[80],
                    nargs='+',
                    help="Image number to do inference")
parser.add_argument("--disp_range",
                    type=int,
                    default=128,
                    help="Search range for disparity")
parser.add_argument("--use_gpu", type=int, default=1, help="Check to use GPU")

args = parser.parse_args()
print('----------------------------------------')
print('FLAGS:')
for arg in vars(args):
    print("'", arg, "'", ": ", getattr(args, arg))
print('----------------------------------------')
print('Inference....')


def load_disp_img(test_num):
    image_path = '%s/disp_%s_0/%06d_10.png' % ('./data_scene_flow/training',
                                               'noc', test_num)
    reader = png.Reader(image_path)
    pngdata = reader.read()
    I_image = np.array(map(np.uint16, pngdata[2]))

    D_image = I_image / 256.0

    return D_image


#################################### Main #####################################
# Trained model file
model_fn = os.path.join(args.checkpoint_dir, args.checkpoint_file)

# Build Test Graph
net = Inference(3, model_fn, args.disp_range, args.use_gpu)
print(net.net)
print('Model Loaded')

for n in args.test_num:
    # Load the images
    l_image_path = os.path.join(args.dataset_dir, 'image_2/%06d_10.png' % n)
    r_image_path = os.path.join(args.dataset_dir, 'image_3/%06d_10.png' % n)

    limg = cv.imread(l_image_path)
    rimg = cv.imread(r_image_path)

    ldisp, rdisp = net.process(limg, rimg)
    print(ldisp.shape, ldisp.dtype)

    cv.imshow('limg', limg)
    cv.imshow('rimg', rimg)
    cv.imshow('ldisp', ldisp / args.disp_range)
    cv.imshow('rdisp', rdisp / args.disp_range)
    k = cv.waitKey(0)
    if k in [27, ord('q')]:
        break

print('Complete!')
