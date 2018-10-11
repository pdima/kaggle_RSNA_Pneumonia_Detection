import numpy as np
import pandas as pd

import math
import os
from torch.utils.data import Dataset
import skimage.color
import skimage.io
from tqdm import tqdm
import utils
import glob
import pickle
import pydicom

import skimage.transform
from collections import namedtuple, defaultdict
from imgaug import augmenters as iaa


import matplotlib.pyplot as plt

from config import *

class DetectionDataset(Dataset):
    def __init__(self, fold, is_training, img_size, images=None, augmentation_level=10, crop_source=1024):
        self.fold = fold
        self.is_training = is_training
        self.img_size = img_size
        self.crop_source = crop_source
        self.augmentation_level = augmentation_level
        self.categories = ['No Lung Opacity / Not Normal', 'Normal', 'Lung Opacity']

        samples = pd.read_csv('../input/stage_1_train_labels.csv')
        samples = samples.merge(pd.read_csv('../input/folds.csv'), on='patientId', how='left')

        if images is None:
            self.images = self.load_images(samples)
        else:
            self.images = images

        if is_training:
            self.samples = samples[samples.fold != fold]
        else:
            self.samples = samples[samples.fold == fold]

        self.patient_ids = list(sorted(self.samples.patientId.unique()))
        self.patient_categories = {}

        self.annotations = defaultdict(list)
        for _, row in self.samples.iterrows():
            patient_id = row['patientId']
            self.patient_categories[patient_id] = self.categories.index(row['class'])
            if row['Target'] > 0:
                x, y, w, h = row.x, row.y, row.width, row.height
                points = np.array([
                    [x, y + h / 3],
                    [x, y + h * 2 / 3],
                    [x + w, y + h / 3],
                    [x + w, y + h * 2 / 3],
                    [x + w / 3, y],
                    [x + w * 2 / 3, y],
                    [x + w / 3, y + h],
                    [x + w * 2 / 3, y + h],
                ])
                self.annotations[patient_id].append(points)

    def load_images(self, samples):
        try:
            images = pickle.load(open(f'{CACHE_DIR}/train_images.pkl', 'rb'))
        except FileNotFoundError:
            os.makedirs(CACHE_DIR, exist_ok=True)
            images = {}
            for patient_id in tqdm(list(sorted(samples.patientId.unique()))):
                dcm_data = pydicom.read_file(f'{TRAIN_DIR}/{patient_id}.dcm')

                img = dcm_data.pixel_array
                # img = skimage.transform.resize(img, (img.shape[0] / 2, img.shape[1] / 2), anti_aliasing=True)
                # img = np.clip(img*255, 0, 255).astype(np.uint8)
                images[patient_id] = img
            pickle.dump(images, open(f'{CACHE_DIR}/train_images.pkl', 'wb'))
        return images

    def load_image(self, patient_id):
        if patient_id in self.images:
            return self.images[patient_id]
        else:
            dcm_data = pydicom.read_file(f'{TRAIN_DIR}/{patient_id}.dcm')
            img = dcm_data.pixel_array
            self.images[patient_id] = img
            return img

    def num_classes(self):
        return 1

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        img = self.load_image(patient_id)

        if self.crop_source != 1024:
            img_source_w = self.crop_source
            img_source_h = self.crop_source
        else:
            img_source_h, img_source_w = img.shape[:2]

        img_h, img_w = img.shape[:2]

        augmentation_sigma = {
            10: dict(scale=0.1, angle=5.0, shear=2.5, gamma=0.2, hflip=False),
            15: dict(scale=0.15, angle=6.0, shear=4.0, gamma=0.2, hflip=np.random.choice([True, False])),
            20: dict(scale=0.15, angle=6.0, shear=4.0, gamma=0.25, hflip=np.random.choice([True, False])),
        }[self.augmentation_level]

        if self.is_training:
            cfg = utils.TransformCfg(
                crop_size=self.img_size,
                src_center_x=img_w/2 + np.random.uniform(-32, 32),
                src_center_y=img_h/2 + np.random.uniform(-32, 32),
                scale_x=self.img_size / img_source_w * (2 ** np.random.normal(0, augmentation_sigma['scale'])),
                scale_y=self.img_size / img_source_h * (2 ** np.random.normal(0, augmentation_sigma['scale'])),
                angle=np.random.normal(0, augmentation_sigma['angle']),
                shear=np.random.normal(0, augmentation_sigma['shear']),
                hflip=augmentation_sigma['hflip'],
                vflip=False
            )
        else:
            cfg = utils.TransformCfg(
                crop_size=self.img_size,
                src_center_x=img_w / 2,
                src_center_y=img_h / 2,
                scale_x=self.img_size / img_source_w,
                scale_y=self.img_size / img_source_h,
                angle=0,
                shear=0,
                hflip=False,
                vflip=False
            )

        crop = cfg.transform_image(img)
        if self.is_training:
            crop = np.power(crop, 2.0 ** np.random.normal(0, augmentation_sigma['gamma']))
            if self.augmentation_level == 20:
                aug = iaa.Sequential(
                    [
                        iaa.Sometimes(0.1, iaa.CoarseSaltAndPepper(p=(0.01, 0.01), size_percent=(0.1, 0.2))),
                        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.0))),
                        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255)))
                    ]
                )
                crop = aug.augment_image(np.clip(np.stack([crop, crop, crop], axis=2) * 255, 0, 255).astype(np.uint8))[:,:,0].astype(np.float32) / 255.0
            if self.augmentation_level == 15:
                aug = iaa.Sequential(
                    [
                        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0.0, 1.0))),
                        iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255)))
                    ]
                )
                crop = aug.augment_image(np.clip(np.stack([crop, crop, crop], axis=2) * 255, 0, 255).astype(np.uint8))[:,:,0].astype(np.float32) / 255.0

        annotations = []
        # print('patient_id', patient_id)

        for annotation in self.annotations[patient_id]:
            points = cfg.transform().inverse(annotation)

            res = np.zeros((1, 5))
            p0 = np.min(points, axis=0)
            p1 = np.max(points, axis=0)

            res[0, 0:2] = p0
            res[0, 2:4] = p1
            res[0, 4] = 0
            annotations.append(res)

        if len(annotations):
            annotations = np.row_stack(annotations)
        else:
            annotations = np.zeros((0, 5))

        sample = {'img': crop, 'annot': annotations, 'scale': 1.0, 'category': self.patient_categories[patient_id]}
        return sample


def check_dataset():
    with utils.timeit_context('load ds'):
        ds = DetectionDataset(fold=0, is_training=True, img_size=512)

    # print(ds.annotations(ds.patient_ids[0]))

    # patient_id = 10056  #ds.patient_ids[0]
    # plt.imshow(ds.images[patient_id])
    #
    # annotation_list = ds.training_samples.loc[[patient_id]]
    #
    # for _, row in annotation_list.iterrows():
    #     plt.plot(
    #         [row[f'p{i}_x'] for i in [1, 2, 3, 4, 1]],
    #         [row[f'p{i}_y'] for i in [1, 2, 3, 4, 1]],
    #         c='y'
    #     )
    # plt.show()

    ds.is_training = False
    plt.imshow(ds[0]['img'])

    plt.figure()
    ds.is_training = True


    for sample in ds:
        plt.cla()
        plt.imshow(sample['img'])
        for annot in sample['annot']:
            p0 = annot[0:2]
            p1 = annot[2:4]

            # print(p0, p1)
            plt.gca().add_patch(plt.Rectangle(p0, width=(p1-p0)[0], height=(p1-p0)[1], fill=False, edgecolor='r', linewidth=2))
        plt.show()


def check_augmentations():
    with utils.timeit_context('load ds'):
        ds = DetectionDataset(fold=0, is_training=True, img_size=512, images={}, augmentation_level=20)

        sample_num = 2

        ds.is_training = False
        plt.imshow(ds[sample_num]['img'])
        for annot in ds[sample_num]['annot']:
            p0 = annot[0:2]
            p1 = annot[2:4]

            # print(p0, p1)
            plt.gca().add_patch(
                plt.Rectangle(p0, width=(p1 - p0)[0], height=(p1 - p0)[1], fill=False, edgecolor='r', linewidth=2))

        plt.figure()
        ds.is_training = True

        for i in range(100):
            sample = ds[sample_num]
            plt.imshow(sample['img'])
            for annot in sample['annot']:
                p0 = annot[0:2]
                p1 = annot[2:4]

                # print(p0, p1)
                plt.gca().add_patch(
                    plt.Rectangle(p0, width=(p1 - p0)[0], height=(p1 - p0)[1], fill=False, edgecolor='r', linewidth=2))
            plt.show()


def check_performance():
    import pytorch_retinanet.dataloader
    import torch
    with utils.timeit_context('load ds'):
        ds = DetectionDataset(fold=0, is_training=True, img_size=512)

    dataloader_train = torch.utils.data.DataLoader(ds, num_workers=16, batch_size=12,
                                  shuffle=True,
                                  collate_fn=pytorch_retinanet.dataloader.collater2d)
    data_iter = tqdm(enumerate(dataloader_train), total=len(dataloader_train))

    with utils.timeit_context('1000 batches:'):
        for iter_num, data in data_iter:
            if iter_num > 1000:
                break


if __name__ == '__main__':
    # check_dataset()
    check_augmentations()
    # check_performance()
