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


class EdlsmNet(nn.Module):
    def __init__(self, nChannel, disp_range, calc_right=False):
        super().__init__()
        self.net = Net(nChannel)
        self.disp_range = disp_range
        self.calc_right = calc_right

    def load_weights(self, fname):
        self.net.load_state_dict(torch.load(fname))

    def forward(self, left_img, right_img):
        l_img = (left_img - left_img.mean()) / (left_img.std())
        r_img = (right_img - right_img.mean()) / (right_img.std())

        left_feat = self.net(l_img)
        right_feat = self.net(r_img)

        # feature shape: 1 64 img_h img_w
        _, _, img_h, img_w = left_feat.size()

        # Output tensor
        unary_vol = torch.zeros((img_h, img_w, self.disp_range),
                                device=left_img.device)
        if self.calc_right:
            right_unary_vol = torch.zeros((img_h, img_w, self.disp_range),
                                          device=left_img.device)
        for d in range(0, self.disp_range):
            # shape: 1 64 img_h img_w-d
            l = left_feat[..., d:img_w]
            r = right_feat[..., 0:img_w - d]

            p = torch.mul(l, r)  # shape: 1 64 img_h img_w-d
            q = torch.sum(p, 1)  # shape: 1 img_h img_w-d

            unary_vol[:, d:img_w, d] = q.data.view(img_h, img_w - d)
            if self.calc_right:
                right_unary_vol[:, 0:img_w - d,
                                d] = q.data.view(img_h, img_w - d)

        _, pred_1 = torch.max(unary_vol, 2)
        if self.calc_right:
            _, pred_2 = torch.max(right_unary_vol, 2)
            return pred_1.view(img_h, img_w), pred_2.view(img_h, img_w)
        return pred_1.view(img_h, img_w)


class Inference:
    def __init__(self,
                 nChannel,
                 model_fn,
                 disp_range,
                 use_gpu=True,
                 calc_right=False):
        self.net = EdlsmNet(nChannel, disp_range, calc_right)
        self.net.load_weights(model_fn)
        self.net.eval()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.net.cuda()

    def preprocess(self, ll_image, rr_image):
        ll_image = torch.from_numpy(ll_image.astype(np.float32)).permute(
            2, 0, 1)
        rr_image = torch.from_numpy(rr_image.astype(np.float32)).permute(
            2, 0, 1)

        # Convert to batch x channel x height x width format
        l_img = l_img.view(1, l_img.size(0), l_img.size(1), l_img.size(2))
        r_img = r_img.view(1, r_img.size(0), r_img.size(1), r_img.size(2))

        return l_img, r_img

    def process(self, l_img, r_img):
        h, w, c = l_img.shape
        l_img = torch.from_numpy(l_img.astype(np.float32)).permute(2, 0, 1)
        r_img = torch.from_numpy(r_img.astype(np.float32)).permute(2, 0, 1)

        l_img = l_img.view(1, c, h, w)
        r_img = r_img.view(1, c, h, w)
        if self.use_gpu:
            l_img = l_img.cuda()
            r_img = r_img.cuda()

        pred = self.net(l_img, r_img)
        if self.net.calc_right:
            return (pred[0].cpu().numpy().astype(np.float32),
                    pred[1].cpu().numpy().astype(np.float32))
        else:
            return (pred.cpu().numpy().astype(np.float32),
                    pred.cpu().numpy().astype(np.float32))


if __name__ == "__main__":
    import time

    left = torch.rand(1, 3, 1242, 375)
    right = torch.rand(1, 3, 1242, 375)

    model = EdlsmNet(3, 128, True)
    model.load_weights('runs/May13_15-36-08_fedor_mirror/edlsm_best.ckpt')
    model.eval()

    left = left.cuda()
    right = right.cuda()
    model.cuda()

    with torch.no_grad():
        for i in range(10):
            tic = time.perf_counter()
            pred = model(left, right)
            toc = time.perf_counter()
            print(f"Inference time {toc - tic:0.4f} s")

    torch.onnx.export(
        model,
        args=(left, right),
        f="edlsm.onnx",
        verbose=False,
        export_params=True,
        #   opset_version=10,
        do_constant_folding=True,
        input_names=['left_img', 'right_img'],
        output_names=['left_disp', 'right_disp'])

    import onnx
    model = onnx.load("edlsm.onnx")

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))
    print(model.graph.input)
    print(model.graph.output)

    # import onnxruntime as ort
    # ort_session = ort.InferenceSession("edlsm.onnx")

    # for i in range(10):
    #     tic = time.perf_counter()
    #     outputs = ort_session.run(
    #         None,
    #         {
    #             "left": left,
    #             "right": right
    #         },
    #     )
    #     toc = time.perf_counter()
    #     print(f"onnxruntime time {toc - tic:0.4f} s")

    # from thop import profile
    # total_ops, total_params = profile(
    #     model,
    #     (
    #         left,
    #         right,
    #     ),
    # )
    # print("{:.4f} MACs(G)\t{:.4f} Params(M)".format(total_ops / (1000**3),
    #                                                 total_params / (1000**2)))
