from __future__ import print_function

import argparse
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms

from src.nets_test import Inference
from src.stereo_metrics import StereoMetrics

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
                    default=[146, 11, 74, 80, 17, 85, 143, 97, 104],
                    nargs='+',
                    help="Image number to do inference")
parser.add_argument("--disp_range",
                    type=int,
                    default=128,
                    help="Search range for disparity")
parser.add_argument("--use_gpu", type=int, default=1, help="Check to use GPU")
parser.add_argument("--out", type=str, default='', help="path store csv")

args = parser.parse_args()
print('----------------------------------------')
print('FLAGS:')
for arg in vars(args):
    print("'", arg, "'", ": ", getattr(args, arg))
print('----------------------------------------')
print('Inference....')

#################################### Main #####################################
# Trained model file
model_fn = os.path.join(args.checkpoint_dir, args.checkpoint_file)

# Build Test Graph
net = Inference(3, model_fn, args.disp_range, args.use_gpu, True)
print(net.net)
print('Model Loaded')

avg = StereoMetrics()
avg.frame = 'Average'
table = []

for n in args.test_num:
    # Load the images
    l_image_path = os.path.join(args.dataset_dir, 'image_2/%06d_10.png' % n)
    r_image_path = os.path.join(args.dataset_dir, 'image_3/%06d_10.png' % n)
    disp_gt_path = os.path.join(args.dataset_dir, 'disp_noc_0/%06d_10.png' % n)

    limg = cv.imread(l_image_path)
    rimg = cv.imread(r_image_path)
    disp_gt = cv.imread(disp_gt_path, cv.IMREAD_UNCHANGED) / 256.0

    ldisp, rdisp = net.process(limg, rimg)

    m = StereoMetrics(disp_gt, ldisp, str(n))
    table.append(m.metrics())
    avg += m

    cv.imshow('limg', limg)
    cv.imshow('rimg', rimg)
    cv.imshow('ldisp', ldisp / args.disp_range)
    cv.imshow('rdisp', rdisp / args.disp_range)
    cv.imshow('disp_gt', disp_gt / args.disp_range)
    cv.imshow('err', m.err / 5)

    k = cv.waitKey(0)
    if k in [27, ord('q')]:
        break

table.append(avg.metrics())
table = pd.DataFrame(table)
pd.set_option('display.max_columns', 20)
print(table)

if args.out:
    table.to_csv(args.out, index=False)
