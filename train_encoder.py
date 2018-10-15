import argparse
import collections
import os
import pickle
import pandas as pd
import pydicom
import skimage.transform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import math
import metric

import pytorch_retinanet.model
import pytorch_retinanet.model_resnet
import pytorch_retinanet.model_se_resnext
import pytorch_retinanet.model_dpn
import pytorch_retinanet.model_pnasnet
import pytorch_retinanet.model_incresv2
import pytorch_retinanet.model_xception
import pytorch_retinanet.model_nasnet_mobile
import pytorch_retinanet.dataloader

import torch.utils.model_zoo as model_zoo
from pretrainedmodels.models import senet

import config
import utils
from config import CROP_SIZE, TEST_DIR
import matplotlib.pyplot as plt

import detection_dataset
from nih_dataset import NihDataset
from logger import Logger


class ModelInfo:
    def __init__(self,
                 factory,
                 args,
                 batch_size,
                 dataset_args,
                 use_sgd=False,
                 img_size=512):
        self.factory = factory
        self.args = args
        self.batch_size = batch_size
        self.dataset_args = dataset_args
        self.img_size = img_size
        self.use_sgd = use_sgd


class SeResNetXt101Encoder(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.num_classes = 15
        self.dropout = dropout

        self.encoder = pytorch_retinanet.model_se_resnext.SeResNetXtEncoder(layers=[3, 4, 23, 3])
        self.encoder.load_state_dict(model_zoo.load_url(
            senet.pretrained_settings['se_resnext101_32x4d']['imagenet']['url'], model_dir='models'), strict=False)

        self.fc15 = nn.Linear(self.encoder.fpn_sizes[-1], self.num_classes)
        self.freeze_bn()

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        img_batch = inputs

        x = torch.stack([img_batch, img_batch, img_batch], dim=1)
        x = self.encoder.layer0(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        out = F.avg_pool2d(x4, x4.shape[2:])
        out = out.view(out.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        x = self.fc15(out)
        x = torch.sigmoid(x)

        return x


MODELS = {
    'se_resnext101_nih': ModelInfo(
        factory=SeResNetXt101Encoder,
        args=dict(dropout=0.5),
        img_size=512,
        batch_size=8,
        dataset_args=dict()
    ),
    'se_resnext101_nih_dr0': ModelInfo(
        factory=SeResNetXt101Encoder,
        args=dict(dropout=0.0),
        img_size=512,
        batch_size=8,
        dataset_args=dict()
    ),
}


def train(model_name, fold, run=None, resume_weights='', resume_epoch=0):
    model_info = MODELS[model_name]

    run_str = '' if run is None or run == '' else f'_{run}'

    checkpoints_dir = f'checkpoints/pretrained/{model_name}{run_str}_fold_{fold}'
    tensorboard_dir = f'../output/tensorboard_pretrained/{model_name}{run_str}_fold_{fold}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    print('\n', model_name, '\n')

    logger = Logger(tensorboard_dir)

    encoder_model = SeResNetXt101Encoder(**model_info.args)
    encoder_model = encoder_model.cuda()

    dataset_train = NihDataset(fold=fold, img_size=model_info.img_size, is_training=True, keep_cache=False)
    dataset_valid = NihDataset(fold=fold, img_size=model_info.img_size, is_training=False, keep_cache=True)

    dataloader_train = DataLoader(dataset_train,
                                  num_workers=16,
                                  batch_size=model_info.batch_size,
                                  shuffle=True,
                                  drop_last=True)

    dataloader_valid = DataLoader(dataset_valid,
                                  num_workers=16,
                                  batch_size=4,
                                  shuffle=False,
                                  drop_last=True)

    encoder_model.training = True

    optimizer = optim.Adam(encoder_model.parameters(), lr=1e-5)
    # optimizer = optim.SGD(encoder_model.parameters(), lr=0.0001, momentum=0.95)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    scheduler_by_epoch = False

    criterion = nn.BCELoss()

    encoder_model.train()

    print('Num training images: {} valid: {}'.format(len(dataset_train), len(dataset_valid)))
    epochs = 32

    # pre-train last layer
    dataloader_pre_train = DataLoader(dataset_train,
                                      num_workers=16,
                                      batch_size=model_info.batch_size,
                                      shuffle=True,
                                      drop_last=True)

    if resume_weights != '':
        encoder_model.load_state_dict(torch.load(resume_weights))

    encoder_model.train()
    encoder_model.freeze_bn()
    encoder_model.freeze_encoder()

    with torch.set_grad_enabled(True):
        for iter_num, data in tqdm(enumerate(dataloader_pre_train), total=1024):
            if iter_num > 1024:
                break
            optimizer.zero_grad()
            labels = data['categories'].cuda().float()

            outputs = encoder_model(data['img'].cuda().float())

            loss = criterion(outputs, labels)

            loss = loss.mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), 0.1)

            optimizer.step()
        del dataloader_pre_train

    encoder_model.unfreeze_encoder()
    for epoch_num in range(resume_epoch+1, epochs):
        encoder_model.train()
        encoder_model.freeze_bn()

        epoch_loss = []

        with torch.set_grad_enabled(True):
            data_iter = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            for iter_num, data in data_iter:
                optimizer.zero_grad()
                labels = data['categories'].cuda().float()

                outputs = encoder_model(data['img'].cuda().float())

                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), 0.1)
                optimizer.step()

                epoch_loss.append(float(loss))
                data_iter.set_description(f'{epoch_num} loss: {np.mean(epoch_loss):1.4f}')

            torch.save(encoder_model.state_dict(), f'{checkpoints_dir}/{model_name}_{epoch_num:03}.pt')

            logger.scalar_summary('loss_train', np.mean(epoch_loss), epoch_num)
            print(np.mean(epoch_loss))

        # validation
        with torch.set_grad_enabled(False):
            encoder_model.eval()

            loss_hist_valid = []

            data_iter = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))
            for iter_num, data in data_iter:
                labels = data['categories'].cuda().float()

                outputs = encoder_model(data['img'].cuda().float())
                loss = criterion(outputs, labels)

                loss_hist_valid.append(float(loss))

                data_iter.set_description(
                    f'{epoch_num} Loss {np.mean(loss_hist_valid):1.4f}')

            logger.scalar_summary('loss_valid', np.mean(loss_hist_valid), epoch_num)
            print(np.mean(loss_hist_valid))

        if scheduler_by_epoch:
            scheduler.step(epoch=epoch_num)
        else:
            scheduler.step(np.mean(loss_hist_valid))
        # if epoch_num % 4 == 0:

    encoder_model.eval()
    torch.save(encoder_model.state_dict(), f'{checkpoints_dir}/{model_name}_final.pt')


def check(model_name, fold, checkpoint):
    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(checkpoint, map_location=device)
    model = model.to(device)
    model.eval()

    dataset_valid = NihDataset(fold=fold, img_size=model_info.img_size, is_training=False)

    dataloader_valid = DataLoader(dataset_valid,
                                  num_workers=1,
                                  batch_size=1,
                                  shuffle=False)

    data_iter = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))
    for iter_num, data in data_iter:
        labels = data['categories'].cuda().float()

        outputs = model(data['img'].cuda().float())

        outputs = outputs.cpu().detach().numpy()

        print(outputs, labels)

        plt.cla()
        plt.imshow(data['img'][0, 0].cpu().detach().numpy())
        plt.show()


# import torchsummary
# m = SeResNetXt101Encoder()
# m.cuda()
# torchsummary.summary(m, (512, 512))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--epoch', type=int, default=-1)

    parser.add_argument('--resume_weights', type=str, default='')
    parser.add_argument('--resume_epoch', type=int, default=-1)

    args = parser.parse_args()
    action = args.action
    model = args.model
    fold = args.fold

    if action == 'train':
        train(model_name=model, run=args.run, fold=args.fold, resume_weights=args.resume_weights, resume_epoch=args.resume_epoch)

    if action == 'check':
        if args.epoch > -1:
            run_str = '' if args.run is None or args.run == '' else f'_{args.run}'
            weights = f'checkpoints/{args.model_name}{run_str}_fold_{fold}/{args.model_name}_{args.epoch:03}.pt'
        else:
            weights = args.weighs

        check(model_name=model, fold=args.fold, checkpoint=weights)

