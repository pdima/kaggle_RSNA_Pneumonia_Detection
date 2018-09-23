import argparse
import collections
import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import skimage.io

import pytorch_retinanet.model
import pytorch_retinanet.model_se_resnext
import pytorch_retinanet.dataloader

import config
from config import CROP_SIZE, TEST_DIR
import matplotlib.pyplot as plt

import detection_dataset
from detection_dataset import DetectionDataset
from logger import Logger


class ModelInfo:
    def __init__(self,
                 factory,
                 args,
                 batch_size,
                 dataset_args,
                 img_size=512):
        self.factory = factory
        self.args = args
        self.batch_size = batch_size
        self.dataset_args = dataset_args
        self.img_size = img_size


MODELS = {
    'resnet34_512': ModelInfo(
        factory=pytorch_retinanet.model.resnet34,
        args=dict(num_classes=1, pretrained=True),
        img_size=512,
        batch_size=8,
        dataset_args=dict()
    ),
    'resnet101_512': ModelInfo(
        factory=pytorch_retinanet.model.resnet101,
        args=dict(num_classes=1, pretrained=True),
        img_size=512,
        batch_size=4,
        dataset_args=dict()
    ),
    'resnet152_512': ModelInfo(
        factory=pytorch_retinanet.model.resnet152,
        args=dict(num_classes=1, pretrained=True),
        img_size=512,
        batch_size=4,
        dataset_args=dict()
    ),
    'se_resnext101_512': ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained=True),
        img_size=512,
        batch_size=3,
        dataset_args=dict()
    ),
    'se_resnext101_512_bs12': ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained=True),
        img_size=512,
        batch_size=12,
        dataset_args=dict()
    ),
    'se_resnext101_256': ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained=True),
        img_size=256,
        batch_size=12,
        dataset_args=dict()
    ),
    'resnet34_256': ModelInfo(
        factory=pytorch_retinanet.model.resnet34,
        args=dict(num_classes=1, pretrained=True),
        img_size=256,
        batch_size=32,
        dataset_args=dict()
    ),
}


def train(model_name, fold, run=None):
    model_info = MODELS[model_name]

    run_str = '' if run is None or run == '' else f'_{run}'

    checkpoints_dir = f'checkpoints/{model_name}{run_str}_fold_{fold}'
    tensorboard_dir = f'../output/tensorboard/{model_name}{run_str}_fold_{fold}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    print('\n', model_name, '\n')

    logger = Logger(tensorboard_dir)

    retinanet = model_info.factory(**model_info.args)
    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()

    dataset_train = DetectionDataset(fold=fold, img_size=model_info.img_size, is_training=True, images={})
    dataset_valid = DetectionDataset(fold=fold, img_size=model_info.img_size, is_training=False, images={})

    dataloader_train = DataLoader(dataset_train,
                                  num_workers=16,
                                  batch_size=model_info.batch_size,
                                  shuffle=True,
                                  collate_fn=pytorch_retinanet.dataloader.collater2d)

    dataloader_valid = DataLoader(dataset_valid,
                                  num_workers=8,
                                  batch_size=model_info.batch_size,
                                  shuffle=False,
                                  collate_fn=pytorch_retinanet.dataloader.collater2d)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True, factor=0.2)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    epochs = 512

    for epoch_num in range(epochs):

        retinanet.train()
        retinanet.module.freeze_bn()
        if epoch_num < 1:
            retinanet.module.freeze_encoder()

        epoch_loss = []
        loss_cls_hist = []
        loss_reg_hist = []

        data_iter = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for iter_num, data in data_iter:
            optimizer.zero_grad()

            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']],
                                                             return_loss=True, return_boxes=False)

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.05)

            optimizer.step()

            loss_cls_hist.append(float(classification_loss))
            loss_reg_hist.append(float(regression_loss))
            epoch_loss.append(float(loss))

            data_iter.set_description(
                f'{epoch_num} cls: {np.mean(loss_cls_hist):1.4f} Reg: {np.mean(loss_reg_hist):1.4f} Running: {np.mean(epoch_loss):1.4f}')

            del classification_loss
            del regression_loss

        logger.scalar_summary('loss_train', np.mean(epoch_loss), epoch_num)
        logger.scalar_summary('loss_train_classification', np.mean(loss_cls_hist), epoch_num)
        logger.scalar_summary('loss_train_regression', np.mean(loss_reg_hist), epoch_num)

        # validation
        with torch.no_grad():
            retinanet.eval()

            loss_hist_valid = []
            loss_cls_hist_valid = []
            loss_reg_hist_valid = []

            data_iter = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))
            for iter_num, data in data_iter:
                res = retinanet.module([data['img'].cuda().float(), data['annot'].cuda().float()],
                                       return_loss=True, return_boxes=True)
                classification_loss, regression_loss, nms_scores, nms_class, transformed_anchors = res

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                loss_hist_valid.append(float(loss))
                loss_cls_hist_valid.append(float(classification_loss))
                loss_reg_hist_valid.append(float(regression_loss))

                data_iter.set_description(
                    f'{epoch_num} cls: {np.mean(loss_cls_hist_valid):1.4f} Reg: {np.mean(loss_reg_hist_valid):1.4f} Total {np.mean(loss_hist_valid):1.4f}')

                del classification_loss
                del regression_loss

            logger.scalar_summary('loss_valid', np.mean(loss_hist_valid), epoch_num)
            logger.scalar_summary('loss_valid_classification', np.mean(loss_cls_hist_valid), epoch_num)
            logger.scalar_summary('loss_valid_regression', np.mean(loss_reg_hist_valid), epoch_num)

        scheduler.step(np.mean(epoch_loss))
        # if epoch_num % 4 == 0:
        torch.save(retinanet.module, f'{checkpoints_dir}/{model_name}_{epoch_num:03}.pt')

    retinanet.eval()
    torch.save(retinanet, f'{checkpoints_dir}/{model_name}_final.pt')


def check(model_name, fold, checkpoint):
    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(checkpoint, map_location=device)
    model = model.to(device)
    model.eval()

    dataset_valid = DetectionDataset(fold=fold, img_size=model_info.img_size, is_training=False,
                                     images={})

    dataloader_valid = DataLoader(dataset_valid,
                                  num_workers=1,
                                  batch_size=1,
                                  shuffle=False,
                                  collate_fn=pytorch_retinanet.dataloader.collater2d)

    data_iter = tqdm(enumerate(dataloader_valid), total=len(dataloader_valid))
    for iter_num, data in data_iter:
        classification_loss, regression_loss, nms_scores, nms_class, transformed_anchors = \
            model([data['img'].to(device).float(), data['annot'].to(device).float()],
                  return_loss=True, return_boxes=True)

        nms_scores = nms_scores.cpu().detach().numpy()
        nms_class = nms_class.cpu().detach().numpy()
        transformed_anchors = transformed_anchors.cpu().detach().numpy()

        print(nms_scores, transformed_anchors.shape)
        print('cls loss:', float(classification_loss), ' reg loss:', float(regression_loss))

        plt.cla()
        plt.imshow(data['img'][0, 0].cpu().detach().numpy())

        gt = data['annot'].cpu().detach().numpy()[0]
        for i in range(gt.shape[0]):
            if np.all(np.isfinite(gt[i])):
                p0 = gt[i, 0:2]
                p1 = gt[i, 2:4]
                plt.gca().add_patch(
                    plt.Rectangle(p0, width=(p1 - p0)[0], height=(p1 - p0)[1], fill=False, edgecolor='b', linewidth=2))

        for i in range(len(nms_scores)):
            nms_score = nms_scores[i]
            if nms_score < 0.1:
                break
            # print(transformed_anchors[i, :])

            p0 = transformed_anchors[i, 0:2]
            p1 = transformed_anchors[i, 2:4]

            color = 'g'
            if nms_score < 0.4:
                color = 'y'
            if nms_score < 0.25:
                color = 'r'

            # print(p0, p1)
            plt.gca().add_patch(plt.Rectangle(p0, width=(p1-p0)[0], height=(p1-p0)[1], fill=False, edgecolor=color, linewidth=2))
            plt.gca().text(p0[0], p0[1], f'{nms_score:.3f}', color=color)  # , bbox={'facecolor': color, 'alpha': 0.5})
        plt.show()

        print(nms_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--weights', type=str, default='')

    args = parser.parse_args()
    action = args.action
    model = args.model
    fold = args.fold

    if action == 'train':
        train(model_name=model, run=args.run, fold=args.fold)

    if action == 'check':
        check(model_name=model, fold=args.fold, checkpoint=args.weights)
