import argparse
import collections
import os
import pickle
import pandas as pd
import pydicom
import skimage.transform

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import metric

import pytorch_retinanet.model
import pytorch_retinanet.model_se_resnext
import pytorch_retinanet.model_dpn
import pytorch_retinanet.model_pnasnet
import pytorch_retinanet.dataloader

import config
import utils
from config import CROP_SIZE, TEST_DIR
import matplotlib.pyplot as plt

import detection_dataset
from detection_dataset import DetectionDataset
from logger import Logger

from train import MODELS, p1p2_to_xywh


def prepare_submission(model_name, run, fold, epoch_num, threshold, submission_name):
    run_str = '' if run is None or run == '' else f'_{run}'
    predictions_dir = f'../output/oof2/{model_name}{run_str}_fold_{fold}'
    os.makedirs(predictions_dir, exist_ok=True)

    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = f'checkpoints/{model_name}{run_str}_fold_{fold}/{model_name}_{epoch_num:03}.pt'
    model = torch.load(checkpoint, map_location=device)
    model = model.to(device)
    model.eval()

    sample_submission = pd.read_csv('../input/stage_1_sample_submission.csv')

    img_size = model_info.img_size
    submission = open(f'../submissions/{submission_name}.csv', 'w')
    submission.write('patientId,PredictionString\n')

    for patient_id in sample_submission.patientId:
        dcm_data = pydicom.read_file(f'{config.TEST_DIR}/{patient_id}.dcm')
        img = dcm_data.pixel_array
        # img = img / 255.0
        img = skimage.transform.resize(img, (img_size, img_size), order=1)
        # utils.print_stats('img', img)

        img_tensor = torch.zeros(1, img_size, img_size, 1)
        img_tensor[0, :, :, 0] = torch.from_numpy(img)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        nms_scores, global_classification, transformed_anchors = \
            model(img_tensor.cuda(), return_loss=False, return_boxes=True)

        scores = nms_scores.cpu().detach().numpy()
        category = global_classification.cpu().detach().numpy()
        boxes = transformed_anchors.cpu().detach().numpy()
        category = np.exp(category[0, 2]) + 0.1 * np.exp(category[0, 0])

        if len(scores):
            scores[scores < scores[0] * 0.5] = 0.0

            # if category > 0.5 and scores[0] < 0.2:
            #     scores[0] *= 2

        # threshold = 0.25
        mask = scores * category * 10 > threshold

        # threshold = 0.5
        # mask = scores * 5 > threshold

        submission_str = ''

        # plt.imshow(dcm_data.pixel_array)

        if np.any(mask):
            boxes_selected = p1p2_to_xywh(boxes[mask])  # x y w h format
            boxes_selected *= 1024.0 / img_size
            scores_selected = scores[mask]

            for i in range(scores_selected.shape[0]):
                x, y, w, h = boxes_selected[i]
                submission_str += f' {scores_selected[i]:.3f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}'
                # plt.gca().add_patch(plt.Rectangle((x,y), width=w, height=h, fill=False, edgecolor='r', linewidth=2))

        print(f'{patient_id},{submission_str}      {category:.2f}')
        submission.write(f'{patient_id},{submission_str}\n')
        # plt.show()


def prepare_submission_multifolds(model_name, run, epoch_num, threshold, submission_name, use_global_cat):
    run_str = '' if run is None or run == '' else f'_{run}'
    models = []

    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions_dir = f'../output/oof2/{model_name}{run_str}_fold_combined'
    os.makedirs(predictions_dir, exist_ok=True)

    for fold in range(4):
        checkpoint = f'checkpoints/{model_name}{run_str}_fold_{fold}/{model_name}_{epoch_num:03}.pt'
        model = torch.load(checkpoint, map_location=device)
        model = model.to(device)
        model.eval()
        models.append(model)

    sample_submission = pd.read_csv('../input/stage_1_sample_submission.csv')

    img_size = model_info.img_size
    submission = open(f'../submissions/{submission_name}.csv', 'w')
    submission.write('patientId,PredictionString\n')

    for patient_id in sample_submission.patientId:
        dcm_data = pydicom.read_file(f'{config.TEST_DIR}/{patient_id}.dcm')
        img = dcm_data.pixel_array
        # img = img / 255.0
        img = skimage.transform.resize(img, (img_size, img_size), order=1)
        # utils.print_stats('img', img)

        img_tensor = torch.zeros(1, img_size, img_size, 1)
        img_tensor[0, :, :, 0] = torch.from_numpy(img)
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        img_tensor = img_tensor.cuda()

        model_raw_results = []
        for model in models:
            model_raw_results.append(model(img_tensor, return_loss=False, return_boxes=False, return_raw=True))

        model_raw_results_mean = []
        for i in range(len(model_raw_results[0])):
            model_raw_results_mean.append(sum(r[i] for r in model_raw_results)/4)

        nms_scores, global_classification, transformed_anchors = models[0].boxes(img_tensor, *model_raw_results_mean)
        # nms_scores, global_classification, transformed_anchors = \
        #     model(img_tensor.cuda(), return_loss=False, return_boxes=True)

        scores = nms_scores.cpu().detach().numpy()
        category = global_classification.cpu().detach().numpy()
        boxes = transformed_anchors.cpu().detach().numpy()
        category = category[0, 2] + 0.1 * category[0, 0]

        if len(scores):
            scores[scores < scores[0] * 0.5] = 0.0

            # if category > 0.5 and scores[0] < 0.2:
            #     scores[0] *= 2

        if use_global_cat:
            mask = scores * category * 10 > threshold
        else:
            mask = scores * 5 > threshold

        submission_str = ''

        # plt.imshow(dcm_data.pixel_array)

        if np.any(mask):
            boxes_selected = p1p2_to_xywh(boxes[mask])  # x y w h format
            boxes_selected *= 1024.0 / img_size
            scores_selected = scores[mask]

            for i in range(scores_selected.shape[0]):
                x, y, w, h = boxes_selected[i]
                submission_str += f' {scores_selected[i]:.3f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}'
                # plt.gca().add_patch(plt.Rectangle((x,y), width=w, height=h, fill=False, edgecolor='r', linewidth=2))

        print(f'{patient_id},{submission_str}      {category:.2f}')
        submission.write(f'{patient_id},{submission_str}\n')
        # plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--from-epoch', type=int, default=1)
    parser.add_argument('--to-epoch', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--use_global_cat', action='store_true')
    parser.add_argument('--submission', type=str, default='')

    args = parser.parse_args()
    action = args.action
    model = args.model
    fold = args.fold

    if action == 'prepare_submission':
        prepare_submission(model_name=model, run=args.run, fold=args.fold, epoch_num=args.epoch,
                           threshold=args.threshold, submission_name=args.submission)

    if action == 'prepare_submission_multifolds':
        with torch.no_grad():
            prepare_submission_multifolds(model_name=model,
                                          run=args.run,
                                          epoch_num=args.epoch,
                                          threshold=args.threshold,
                                          submission_name=args.submission,
                                          use_global_cat=args.use_global_cat
                                          )
