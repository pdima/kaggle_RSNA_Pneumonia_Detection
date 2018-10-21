import argparse
import os
import pickle

import numpy as np
import pandas as pd
import pydicom
import skimage.transform
import torch
import cv2
import utils

import config
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


def prepare_submission_multifolds(model_name, run, epoch_nums, threshold, submission_name, use_global_cat):
    run_str = '' if run is None or run == '' else f'_{run}'
    models = []

    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions_dir = f'../output/oof2/{model_name}{run_str}_fold_combined'
    os.makedirs(predictions_dir, exist_ok=True)

    for epoch_num in epoch_nums:
        for fold in range(4):
            checkpoint = f'checkpoints/{model_name}{run_str}_fold_{fold}/{model_name}_{epoch_num:03}.pt'
            print('load', checkpoint)
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
            model_raw_results_mean.append(sum(r[i] for r in model_raw_results)/len(models))

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


def prepare_test_predictions(model_name, run, epoch_num):
    run_str = '' if run is None or run == '' else f'_{run}'
    models = []

    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)

    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = model_info.img_size
    # print('epoch', epoch_num)

    for fold in range(4):
        print('fold', fold)
        output_dir = f'{config.TEST_PREDICTIONS_DIR}/{model_name}{run_str}_fold_{fold}/{epoch_num:03}/'
        os.makedirs(output_dir, exist_ok=True)

        checkpoint = f'checkpoints/{model_name}{run_str}_fold_{fold}/{model_name}_{epoch_num:03}.pt'
        print('load', checkpoint)
        model = torch.load(checkpoint, map_location=device)
        model = model.to(device)
        model.eval()
        models.append(model)

        for patient_id in sample_submission.patientId:
            dcm_data = pydicom.read_file(f'{config.TEST_DIR}/{patient_id}.dcm')
            img = dcm_data.pixel_array
            # img = img / 255.0
            if img_size != 1024:
                # img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                # img = img.astype(np.float32) / 255.0
                img = skimage.transform.resize(img, (img_size, img_size), order=1)
            else:
                img = img.astype(np.float32) / 255.0
            # utils.print_stats('img', img)

            img_tensor = torch.zeros(1, img_size, img_size, 1)
            img_tensor[0, :, :, 0] = torch.from_numpy(img)
            img_tensor = img_tensor.permute(0, 3, 1, 2)
            img_tensor = img_tensor.cuda()

            model_raw_results = model(img_tensor, return_loss=False, return_boxes=False, return_raw=True)
            # discard last item - anchors
            model_raw_results_cpu = [r.cpu().detach().numpy() for r in model_raw_results[:-1]]

            pickle.dump(model_raw_results_cpu, open(f'{output_dir}/{patient_id}.pkl', 'wb'))


def prepare_submission_from_saved(model_name, run, epoch_nums, threshold, submission_name, use_global_cat, size_scale):
    run_str = '' if run is None or run == '' else f'_{run}'

    model_info = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = f'checkpoints/{model_name}{run_str}_fold_0/{model_name}_{epoch_nums[0]:03}.pt'
    print('load', checkpoint)
    model = torch.load(checkpoint, map_location=device)
    model = model.to(device)
    model.eval()

    img_size = model_info.img_size
    img_tensor = torch.zeros(1, img_size, img_size, 1).permute(0, 3, 1, 2).to(device)

    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)

    img_size = model_info.img_size
    submission = open(f'../submissions/{submission_name}.csv', 'w')
    submission.write('patientId,PredictionString\n')

    anchors = model.anchors(img_tensor)

    for patient_id in sample_submission.patientId:
        regression_results = []
        classification_results = []
        global_classification_results = []
        # anchors = []

        for epoch_num in epoch_nums:
            for fold in range(4):
                saved_dir = f'{config.TEST_PREDICTIONS_DIR}/{model_name}{run_str}_fold_{fold}/{epoch_num:03}/'
                model_raw_result = pickle.load(open(f'{saved_dir}/{patient_id}.pkl', 'rb'))
                # model_raw_result = [torch.from_numpy(r).to(device) for r in model_raw_result_numpy]

                regression_results.append(model_raw_result[0])
                classification_results.append(model_raw_result[1])
                global_classification_results.append(model_raw_result[2])
                # anchors = model_raw_result[3]  # anchors all the same

        regression_results = np.concatenate(regression_results, axis=0)

        regression_results_pos = regression_results[:, :, :2]
        regression_results_pos = np.mean(regression_results_pos, axis=0, keepdims=True)

        regression_results_size = regression_results[:, :, 2:]
        regression_results_size_p80 = np.percentile(regression_results_size, q=80, axis=0, keepdims=True)
        regression_results_size = np.percentile(regression_results_size, q=20, axis=0, keepdims=True)

        regression_results_size += (regression_results_size - regression_results_size_p80) * size_scale

        regression_results = np.concatenate([regression_results_pos, regression_results_size], axis=2).astype(np.float32)

        # regression_results = np.mean(regression_results, axis=0, keepdims=True)

        classification_results = np.concatenate(classification_results, axis=0)
        classification_results = np.mean(classification_results, axis=0, keepdims=True)

        global_classification_results = np.concatenate(global_classification_results, axis=0)
        global_classification_results = np.mean(global_classification_results, axis=0, keepdims=True)

        # model_raw_results_mean = []
        # for i in range(len(model_raw_results[0])):
        #     model_raw_results_mean.append(sum(r[i] for r in model_raw_results)/len(model_raw_results))

        nms_scores, global_classification, transformed_anchors = model.boxes(
            img_tensor,
            torch.from_numpy(regression_results).to(device),
            torch.from_numpy(classification_results).to(device),
            torch.from_numpy(global_classification_results).to(device),
            anchors
        )
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
            mask = ((scores * category) ** 0.5) * 5 > threshold
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
    submission.close()
    check_submission_stat(submission_name)


def check_submission_stat(sub_name):
    all_rects = []
    if sub_name.endswith('.csv'):
        sub_name = sub_name[:-4]

    nb_non_empy = 0

    for line in open(f'../submissions/{sub_name}.csv', 'r'):
        if line.startswith('patientId'):
            continue

        patient_id, sub = line.split(',')
        items = [float(i) for i in sub.split()]
        nb_rects = len(items) // 5
        for rect_id in range(nb_rects):
            all_rects.append(items[rect_id*5: rect_id*5+5])
            # prob, x, y, w, h = items[rect_id*5: rect_id*5+5]

        if nb_rects > 0:
            nb_non_empy += 1

    all_rects = np.array(all_rects)

    w = all_rects[:, -2]
    h = all_rects[:, -1]

    print(f'w mean {np.mean(w):.1f} w med {np.median(w):.1f}  h mean {np.mean(h):.1f} h med {np.median(h):.1f}  '
          f'area mean {np.mean(w*h):.1f} area med {np.median(w*h):.1f}   {nb_non_empy} {sub_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--epoch', type=int, nargs='+')
    parser.add_argument('--from-epoch', type=int, default=1)
    parser.add_argument('--to-epoch', type=int, default=100)
    parser.add_argument('--size_perc', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--size_scale', type=float, default=0.9)
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
                                          epoch_nums=args.epoch,
                                          threshold=args.threshold,
                                          submission_name=args.submission,
                                          use_global_cat=args.use_global_cat
                                          )
    if action == 'prepare_test_predictions':
        for epoch_num in args.epoch:
            prepare_test_predictions(model_name=model, run=args.run, epoch_num=epoch_num)

    if action == 'prepare_submission_from_saved':
        with torch.no_grad():
            prepare_submission_from_saved(model_name=model,
                                          run=args.run,
                                          epoch_nums=args.epoch,
                                          threshold=args.threshold,
                                          submission_name=args.submission,
                                          use_global_cat=args.use_global_cat,
                                          size_scale=args.size_scale
                                          )

    if action == 'check_stat':
        check_submission_stat(args.submission)
