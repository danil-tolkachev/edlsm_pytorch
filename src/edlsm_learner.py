from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import cv2 as cv
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
import shutil
import json

from .nets import Net
from .data_Loader import dataLoader
from .nets_test import Inference
from .stereo_metrics import StereoMetrics


def dict_to_md(d):
    txt = []
    txt.append('|Key|Value|')
    txt.append('|---|---|')
    for k, v in d.items():
        txt.append(f'|{k}|{v}|')
    return '\n'.join(txt)


def calc_metrics(net, loader, directory, frame):
    avg = StereoMetrics()
    avg.frame = frame

    for n in tqdm(loader.img_set):
        l_image_path = os.path.join(directory, 'image_2/%06d_10.png' % n)
        r_image_path = os.path.join(directory, 'image_3/%06d_10.png' % n)
        disp_gt_path = os.path.join(directory, 'disp_noc_0/%06d_10.png' % n)

        limg = cv.imread(l_image_path)
        rimg = cv.imread(r_image_path)
        disp_gt = cv.imread(disp_gt_path, cv.IMREAD_UNCHANGED) / 256.0

        ldisp, _ = net.process(limg, rimg, right=False)

        m = StereoMetrics(disp_gt, ldisp, str(n))
        avg += m
    return avg.metrics()

class edlsmLearner(object):
    def __init__(self):
        pass

    def loss_function(self, x3, t, w):
        # Three pixel error
        error = 0
        for i in range(x3.size(0)):
            sc = x3[i,t[i][0]-2:t[i][0]+2+1]
            loss_sample = torch.mul(sc, w).sum()

            error = error - loss_sample

        return error

    def train(self, opt):
        # Load Flags
        self.opt = opt

        # Data Loader
        train_loader = dataLoader(opt.directory,
                                  opt.psz,
                                  opt.half_range,
                                  mode='Train',
                                  train_split_size=opt.train_split_size)
        train_gen = train_loader.gen_data_batch(batch_size=opt.batch_size)

        val_loader = dataLoader(opt.directory,
                                opt.psz,
                                opt.half_range,
                                mode='Test',
                                train_split_size=opt.train_split_size)

        # Build training graph
        model = Net(3, opt.half_range*2+1)
        pxl_wghts = opt.pxl_wghts/np.sum(opt.pxl_wghts)
        pxl_wghts = Variable(torch.Tensor(pxl_wghts))
        if opt.gpu:
            model = model.cuda()
            pxl_wghts = pxl_wghts.cuda()
        model.train()

        # Check if training has to be continued
        if opt.continue_train:
            if opt.init_checkpoint_file is None:
                print('Enter a valid checkpoint file')
            else:
                load_model = os.path.join(opt.checkpoint_dir, opt.init_checkpoint_file)
            print("Resume training from previous checkpoint: %s" % opt.init_checkpoint_file)
            model.load_state_dict(torch.load(load_model))

        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=opt.l_rate, weight_decay=opt.l2)

        writer = SummaryWriter()
        best_metrics = None
        writer.add_text('Options', dict_to_md(vars(opt)))

        # Begin Training
        for step in range(opt.start_step, opt.max_steps, opt.save_latest_freq):
            print('Train:')
            for _ in tqdm(range(opt.save_latest_freq)):
                # Sample batch data
                left_batch, right_batch, t_batch = next(train_gen)

                # Zero the gradient buffers
                optimizer.zero_grad()

                # Convert to cuda if GPU available
                if opt.gpu:
                    left_batch  = left_batch.cuda()
                    right_batch = right_batch.cuda()
                    t_batch     = t_batch.cuda()

                # Forward pass
                x1, x2, x3 = model(Variable(left_batch), Variable(right_batch))

                # Compute loss
                loss = self.loss_function(x3, t_batch, pxl_wghts)

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

            model_name = 'edlsm_latest.ckpt'
            checkpoint_path = os.path.join(writer.log_dir, model_name)
            torch.save(model.state_dict(), checkpoint_path)

            step += opt.save_latest_freq
            print('Step Loss: ',
                  loss.data.cpu().numpy() / opt.batch_size, ' at iteration: ',
                  step)
            writer.add_scalar('Loss/train',
                              loss.data.cpu().numpy() / opt.batch_size, step)

            net = Inference(3, checkpoint_path, 128, True)

            m = calc_metrics(
                net, train_loader, opt.directory,
                f'Train metrics at step {step} epoch {train_loader.epoch}')
            print('Train metrics:')
            pprint(m)
            writer.add_scalar('Train/Err1px', m['Err1px'], step)
            writer.add_scalar('Train/Err2px', m['Err2px'], step)
            writer.add_scalar('Train/Err3px', m['Err3px'], step)

            m = calc_metrics(
                net, val_loader, opt.directory,
                f'Validation metrics at step {step} epoch {train_loader.epoch}'
            )
            print('Validation metrics:')
            pprint(m)
            writer.add_scalar('Validation/Err1px', m['Err1px'], step)
            writer.add_scalar('Validation/Err2px', m['Err2px'], step)
            writer.add_scalar('Validation/Err3px', m['Err3px'], step)

            if best_metrics is None or best_metrics['Err3px'] > m['Err3px']:
                best_metrics = m
                best_checkpoint_path = os.path.join(
                    writer.log_dir, 'edlsm_best.ckpt')
                shutil.copy(checkpoint_path, best_checkpoint_path)
                print('Update', best_checkpoint_path)
                metrics_path = os.path.join(writer.log_dir, 'best_metrics.json')
                json.dump(best_metrics, open(metrics_path, 'w'), indent=2)

        writer.add_text('Best metrics', dict_to_md(best_metrics))
        writer.close()

        # Save the latest
        print("Training complete!")
        print("Best metrics:")
        pprint(best_metrics)
