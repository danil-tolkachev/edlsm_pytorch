from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.nets_test import Inference

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
                    default=80,
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


# Useful functions
def load_and_resize_l_and_r_image(test_num):
    # Load the image
    l_image_path = os.path.join(args.dataset_dir,
                                'image_2/%06d_10.png' % (test_num))
    r_image_path = os.path.join(args.dataset_dir,
                                'image_3/%06d_10.png' % (test_num))
    ll_image1 = Image.open(l_image_path)
    ll_image1 = ll_image1.convert('RGB')
    rr_image1 = Image.open(r_image_path)
    rr_image1 = rr_image1.convert('RGB')

    ll_image1 = np.array(ll_image1)
    rr_image1 = np.array(rr_image1)

    ll_image = 255 * transforms.ToTensor()(ll_image1)
    rr_image = 255 * transforms.ToTensor()(rr_image1)

    return ll_image, rr_image, ll_image1, rr_image1


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

# Load the images
ll_image, rr_image, ll_image1, rr_image1 = load_and_resize_l_and_r_image(
    args.test_num)

pred_disp1, pred_disp2 = net.process(ll_image, rr_image)

# Display the images
plt.subplot(411)
plt.imshow(ll_image1)
plt.title('Left Image')
plt.axis('off')
plt.subplot(412)
plt.imshow(rr_image1)
plt.title('Right Image')
plt.axis('off')
plt.subplot(413)
plt.imshow(pred_disp1, cmap='gray')
plt.title('Predicted Disparity')
plt.axis('off')
plt.subplot(414)
plt.imshow(pred_disp2, cmap='gray')
plt.title('Right Disparity')
plt.axis('off')
plt.show()

print('Complete!')
