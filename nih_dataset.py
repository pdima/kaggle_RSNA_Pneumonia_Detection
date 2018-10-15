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
from PIL import Image

import skimage.transform
from collections import namedtuple, defaultdict
from imgaug import augmenters as iaa


import matplotlib.pyplot as plt

from config import *


class NihDataset(Dataset):
    def __init__(self, fold, is_training, img_size, keep_cache=False, verbose=False):
        self.fold = fold
        self.is_training = is_training
        self.img_size = img_size
        self.keep_cache = keep_cache
        self.verbose = verbose
        self.categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
                           'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
                           'Pneumothorax']

        samples = pd.read_csv('../input/nih_folds.csv')

        self.images = {}

        if is_training:
            self.samples = samples[samples.fold != fold]
        else:
            self.samples = samples[samples.fold == fold]

        self.patient_ids = list(sorted(self.samples.fn))
        self.patient_categories = {}

        print(samples.shape, self.samples.shape, len(self.patient_ids))

        self.annotations = defaultdict(list)
        for _, row in self.samples.iterrows():
            patient_id = row['fn']
            categories = row['Finding Labels'].split('|')
            self.patient_categories[patient_id] = np.array([c in categories for c in self.categories])

    def load_image(self, patient_id):
        if patient_id in self.images:
            return self.images[patient_id]
        else:
            img = np.array(Image.open('../data/nih/images/' + patient_id))
            if len(img.shape) > 2:
                img = img[:, :, 0]
            # print(img.shape)
            if self.keep_cache:
                self.images[patient_id] = img
            return img

    def num_classes(self):
        return 15

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        if self.verbose:
            print(patient_id)

        img = self.load_image(patient_id)

        img_source_h, img_source_w = img.shape[:2]
        img_h, img_w = img.shape[:2]

        if self.is_training:
            cfg = utils.TransformCfg(
                crop_size=self.img_size,
                src_center_x=img_w/2 + np.random.uniform(-32, 32),
                src_center_y=img_h/2 + np.random.uniform(-32, 32),
                scale_x=self.img_size / img_source_w * (2 ** np.random.normal(0, 0.25)),
                scale_y=self.img_size / img_source_h * (2 ** np.random.normal(0, 0.25)),
                angle=np.random.normal(0, 8.0),
                shear=np.random.normal(0, 4.0),
                hflip=np.random.choice([True, False]),
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
            crop = np.power(crop, 2.0 ** np.random.normal(0, 0.2))
            aug = iaa.Sequential(
                [
                    iaa.Sometimes(0.1, iaa.CoarseSaltAndPepper(p=(0.01, 0.01), size_percent=(0.1, 0.2))),
                    iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0.0, 2.0))),
                    iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255)))
                ]
            )
            crop = aug.augment_image(np.clip(np.stack([crop, crop, crop], axis=2) * 255, 0, 255).astype(np.uint8))[:,:,0].astype(np.float32) / 255.0

        # soft_label = 1e-4
        # labels = self.patient_categories[patient_id] * (1.0 - soft_label * 2) + soft_label
        labels = self.patient_categories[patient_id].astype(np.float32)
        sample = {'img': crop, 'categories': labels}
        return sample


def check_dataset():
    with utils.timeit_context('load ds'):
        ds = NihDataset(fold=0, is_training=True, img_size=512, verbose=True)

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
        print(sample['categories'])
        print(np.array(ds.categories)[sample['categories'] > 0.5])
        plt.cla()
        plt.imshow(sample['img'])
        plt.show()


def check_augmentations():
    with utils.timeit_context('load ds'):
        ds = NihDataset(fold=0, is_training=True, img_size=512)

        sample_num = 2

        ds.is_training = False
        plt.imshow(ds[sample_num]['img'])

        plt.figure()
        ds.is_training = True

        for i in range(100):
            sample = ds[sample_num]
            utils.print_stats('img', sample['img'])
            plt.imshow(sample['img'])
            plt.show()



if __name__ == '__main__':
    check_dataset()
    # check_augmentations()
    # check_performance()
