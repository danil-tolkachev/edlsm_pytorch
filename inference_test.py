from __future__ import print_function

import argparse

import cv2 as cv

from src.nets_test import Inference
from src.postprocess import lr_check

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, help="checkpoint file name to load")
parser.add_argument("--left_img",
                    type=str,
                    default='data_scene_flow/training/image_2/000000_10.png')
parser.add_argument("--right_img",
                    type=str,
                    default='data_scene_flow/training/image_3/000000_10.png')
parser.add_argument("--disp_range",
                    type=int,
                    default=128,
                    help="Search range for disparity")
parser.add_argument("--use_gpu", type=int, default=1, help="Check to use GPU")

args = parser.parse_args()

# Build Test Graph
net = Inference(3, args.weights, args.disp_range, args.use_gpu)

limg = cv.imread(args.left_img)
rimg = cv.imread(args.right_img)

ldisp, rdisp = net.process(limg, rimg)
lr_check(ldisp, rdisp)

cv.imshow('limg', limg)
cv.imshow('rimg', rimg)
cv.imshow('ldisp', ldisp / args.disp_range)
cv.imshow('rdisp', rdisp / args.disp_range)

k = cv.waitKey(0)
