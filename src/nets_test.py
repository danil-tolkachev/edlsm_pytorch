import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Refrenced from: https://github.com/vijaykumar01/stereo_matching
class Net(nn.Module):
    def __init__(self, nChannel):
        super(Net, self).__init__()
        # Perform 18 pixel padding on the image on all sides.
        self.pad = nn.ReflectionPad2d(18)

        # first conv layer: 32 filters of size 5x5
        self.conv1 = nn.Conv2d(nChannel, 32, 5)
        # first batch normalization layer
        self.batchnorm1 = nn.BatchNorm2d(32, 1e-3)

        # second conv layer: 32 filters of size 5x5
        self.conv2 = nn.Conv2d(32, 32, 5)
        # second normalization layer
        self.batchnorm2 = nn.BatchNorm2d(32, 1e-3)

        # third conv layer: 64 filters of size 5x5
        self.conv3 = nn.Conv2d(32, 64, 5)
        # third batch normalization layer
        self.batchnorm3 = nn.BatchNorm2d(64, 1e-3)

        # fourth conv layer: 64 filters of size 5x5
        self.conv4 = nn.Conv2d(64, 64, 5)
        # fourth batch normalization layer
        self.batchnorm4 = nn.BatchNorm2d(64, 1e-3)

        # fifth conv layer: 64 filters of size 5x5
        self.conv5 = nn.Conv2d(64, 64, 5)
        # fifth batch normalization layer
        self.batchnorm5 = nn.BatchNorm2d(64, 1e-3)

        # sixth conv layer: 64 filters of size 5x5
        self.conv6 = nn.Conv2d(64, 64, 5)
        # sixth batch normalization layer
        self.batchnorm6 = nn.BatchNorm2d(64, 1e-3)

        # seventh conv layer: 64 filters of size 5x5
        self.conv7 = nn.Conv2d(64, 64, 5)
        # seventh batch normalization layer
        self.batchnorm7 = nn.BatchNorm2d(64, 1e-3)

        # eighth conv layer: 64 filters of size 5x5
        self.conv8 = nn.Conv2d(64, 64, 5)
        # eigth batch normalization layer
        self.batchnorm8 = nn.BatchNorm2d(64, 1e-3)

        # ninth conv layer: 64 filters of size 5x5
        self.conv9 = nn.Conv2d(64, 64, 5)
        # ninth batch normalization layer
        self.batchnorm9 = nn.BatchNorm2d(64, 1e-3)

    def forward(self, x):
        with torch.no_grad():
            x = self.pad(x)
            x = self.conv1(x)
            x = F.relu(self.batchnorm1(x))

            x = self.conv2(x)
            x = F.relu(self.batchnorm2(x))

            x = self.conv3(x)
            x = F.relu(self.batchnorm3(x))

            x = self.conv4(x)
            x = F.relu(self.batchnorm4(x))

            x = self.conv5(x)
            x = F.relu(self.batchnorm5(x))

            x = self.conv6(x)
            x = F.relu(self.batchnorm6(x))

            x = self.conv7(x)
            x = F.relu(self.batchnorm7(x))

            x = self.conv8(x)
            x = F.relu(self.batchnorm8(x))

            x = self.conv9(x)
            x = self.batchnorm9(x)

        return x


class Inference:
    def __init__(self, nChannel, model_fn, disp_range, use_gpu=True):
        self.net = Net(nChannel)
        self.net.load_state_dict(torch.load(model_fn))
        self.net.eval()
        self.use_gpu = use_gpu
        self.disp_range = disp_range
        if use_gpu:
            self.net = self.net.cuda()

    def preprocess(self, ll_image, rr_image):
        ll_image = torch.from_numpy(ll_image.astype(np.float32)).permute(
            2, 0, 1)
        rr_image = torch.from_numpy(rr_image.astype(np.float32)).permute(
            2, 0, 1)

        # Normalize images. All the patches used for training were normalized.
        l_img = (ll_image - ll_image.mean()) / (ll_image.std())
        r_img = (rr_image - rr_image.mean()) / (rr_image.std())

        # Convert to batch x channel x height x width format
        l_img = l_img.view(1, l_img.size(0), l_img.size(1), l_img.size(2))
        r_img = r_img.view(1, r_img.size(0), r_img.size(1), r_img.size(2))

        if self.use_gpu:
            l_img = l_img.cuda()
            r_img = r_img.cuda()

        return l_img, r_img

    def calc_features(self, l_img, r_img):
        # Forward pass. extract deep features
        left_feat = self.net(Variable(l_img, requires_grad=False))
        # forward pass right image
        right_feat = self.net(Variable(r_img, requires_grad=False))

        return left_feat, right_feat

    def calc_disparity(self, left_feat, right_feat, right=True):
        # feature shape: 1 64 img_h img_w
        _, _, img_h, img_w = left_feat.size()

        # Output tensor
        unary_vol = torch.Tensor(img_h, img_w, self.disp_range).zero_()
        if right:
            right_unary_vol = torch.Tensor(img_h, img_w,
                                           self.disp_range).zero_()

        for d in range(0, self.disp_range):
            # shape: 1 64 img_h img_w-d
            l = left_feat[..., d:img_w]
            r = right_feat[..., 0:img_w - d]

            p = torch.mul(l, r) # shape: 1 64 img_h img_w-d
            q = torch.sum(p, 1) # shape: 1 img_h img_w-d

            unary_vol[:, d:img_w, d] = q.data.view(q.data.size(1),
                                                   q.data.size(2))
            if right:
                right_unary_vol[:, 0:img_w - d,
                                d] = q.data.view(q.data.size(1),
                                                 q.data.size(2))

        max_disp1, pred_1 = torch.max(unary_vol, 2)
        pred_disp1 = pred_1.view(img_h, img_w)

        if right:
            max_disp2, pred_2 = torch.max(right_unary_vol, 2)
            pred_disp2 = pred_2.view(img_h, img_w)
        else:
            pred_disp2 = pred_disp1

        return pred_disp1, pred_disp2

    def process(self, ll_image, rr_image, right=True):
        l_img, r_img = self.preprocess(ll_image, rr_image)
        left_feat, right_feat = self.calc_features(l_img, r_img)
        pred_disp1, pred_disp2 = self.calc_disparity(left_feat,
                                                     right_feat,
                                                     right=right)
        return (pred_disp1.numpy().astype(np.float32),
                pred_disp2.numpy().astype(np.float32))
