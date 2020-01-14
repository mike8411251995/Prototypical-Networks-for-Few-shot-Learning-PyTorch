import os
from os.path import join as pjoin
from pathlib import Path
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import re

from PIL import Image
from tqdm import tqdm
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
from ptsemseg.loader.oslsm import ss_datalayer
from ptsemseg.loader import pascalVOCLoader
import yaml
import cv2

from dataloaders import custom_transforms_image as tr_image
from dataloaders import custom_transforms as tr_sample

class pascalVOC5iLoader(pascalVOCLoader):
    """Data loader for the Pascal VOC 5i Few-shot semantic segmentation dataset.
    """

    def __init__(
        self,
        root,
        args,
        deeplab_model,
        inverse=False,
        split="val",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        n_classes=15,
        fold=0,
        binary=False,
        k_shot=1
    ):
        super(pascalVOC5iLoader, self).__init__(root, split=split,
                                          is_transform=is_transform, img_size=img_size,
                                          augmentations=augmentations, img_norm=img_norm,
                                          n_classes=n_classes)

        with open('ptsemseg/loader/oslsm/profile.txt', 'r') as f:
            profile = str(f.read())
            profile = self.convert_d(profile)

        profile['pascal_path'] = self.root
        profile['areaRng'][1] = float('Inf')

        pascal_lbls = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                       'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

        profile['pascal_cats'] = []
        for i in range(fold*5+1, (fold+1)*5+1):
            profile['pascal_cats'].append(pascal_lbls[i])

        profile['k_shot'] = k_shot
        profile_copy = profile.copy()
        profile_copy['first_label_params'].append(('original_first_label', 1.0, 0.0))
        profile_copy['deploy_mode'] = True

        dbi = ss_datalayer.DBInterface(profile, fold=fold, binary=binary)
        self.PLP = ss_datalayer.PairLoaderProcess(None, None, dbi, profile_copy)

        self.inverse = inverse
        if self.inverse:
            self.oslsm_files = self.parse_file('ptsemseg/loader/imgs_paths_%d_%d_inv.txt'%(fold, k_shot), k_shot)            
        else:
            self.oslsm_files = self.parse_file('ptsemseg/loader/imgs_paths_%d_%d.txt'%(fold, k_shot), k_shot)
        
        self.prefix_lbl = 'SegmentationClass/pre_encoded/'

        self.args = args
        self.deeplab_model = deeplab_model
        deeplab_model_name = re.search('pascal/(.*)/model_best', self.args.resume).group(1)
        self.prefix_feat = 'PixelwiseFeatures/{}/'.format(deeplab_model_name)
        
        self.current_fold = fold


    def parse_file(self, pth_txt, k_shot):
        files = []
        pair = []
        support = []
        f = open(pth_txt, 'r')
        count = 0
        for line in f:
            if count == (k_shot+1)*2:
                pair.append(line.split(' ')[-1].strip())
                files.append(pair)
                count = -1
            elif count < k_shot:
                support.append(line.strip())
            elif count < k_shot+1:
                pair = [support, line.strip()]
                support = []
            count += 1
        return files

    def convert_d(self, string):
        s = string.replace("{" ,"")
        finalstring = s.replace("}" , "")
        list = finalstring.split(";")

        dictionary ={}
        for i in list:
            keyvalue = i.split(":")
            m = eval(keyvalue[0])
            dictionary[m] = eval(keyvalue[1])
        return dictionary

    def __len__(self):
        return 3000 if self.inverse else 1000
        #len(self.PLP.db_interface.db_items)

    def map_labels(self, lbl, cls_idx):
        # ignore_classes = range(self.current_fold*5+1, (self.current_fold+1)*5+1)
        # class_count = 0
        # temp_lbl = lbl.copy()
        # for c in range(21):
        #     if c not in ignore_classes:
        #         temp_lbl[lbl == c] = class_count
        #         class_count += 1
        #     elif c in ignore_classes and c!=cls_idx:
        #         temp_lbl[lbl == c] = 0
        # temp_lbl[lbl==cls_idx] = 16
        # return temp_lbl
        temp_lbl = lbl.copy()
        temp_lbl[lbl==cls_idx] = 1
        temp_lbl[lbl!=cls_idx] = 0
        return temp_lbl

    def __getitem__(self, index):
        pair = self.oslsm_files[index]

        qry_img_orig = Image.open(self.root+pair[1]) # before transform

        qry_lbl_orig = cv2.imread(self.root+pair[1].replace('JPEGImages', self.prefix_lbl).replace('jpg', 'png') , 0)
        qry_lbl_orig = np.asarray(qry_lbl_orig, dtype=np.int32)
        qry_lbl_orig = self.map_labels(qry_lbl_orig, int(pair[-1]))
        qry_lbl_orig = Image.fromarray(qry_lbl_orig)

        qry_transformed = self.transform_sample({'image': qry_img_orig, 'label': qry_lbl_orig})
        qry_img = qry_transformed['image']
        qry_lbl = qry_transformed['label']

        qry_feat_file = self.root+pair[1].replace('JPEGImages', self.prefix_feat).replace('jpg', 'pt')
        try:
            qry_feat = torch.load(qry_feat_file)
        except FileNotFoundError:
            qry_feat = self.deeplab_model.module.extract_pixel_feature(qry_img[None, :].to('cuda')).detach().cpu()
            Path(qry_feat_file).parent.mkdir(parents=True, exist_ok=True)
            torch.save(qry_feat, qry_feat_file)
        qry_lbl = F.interpolate(qry_lbl[None, None, :], size=qry_feat.shape[-1])
        qry_feat = qry_feat.reshape(qry_feat.shape[1], -1)
        qry_lbl = qry_lbl.reshape(-1)

        spt_imgs_orig = [] # before transform
        spt_lbls_orig = [] # before transform
        spt_imgs = []
        spt_feats = []
        spt_lbls = []
        spt_masked = []

        for j in range(len(pair[0])):
            spt_imgs_orig_j = Image.open(self.root+pair[0][j])
            spt_imgs_orig.append(spt_imgs_orig_j)

            spt_lbls_orig_j = cv2.imread(self.root+pair[0][j].replace('JPEGImages', self.prefix_lbl).replace('jpg', 'png') , 0)
            spt_lbls_orig_j = np.asarray(spt_lbls_orig_j, dtype=np.int32)

            spt_binary_j = (spt_lbls_orig_j == int(pair[-1]))
            spt_binary_j = np.stack([spt_binary_j for _ in range(3)], axis=0).transpose(1, 2, 0)
            spt_masked_j = np.multiply(np.asarray(spt_imgs_orig_j), spt_binary_j)
            spt_masked_j = self.transform_image(Image.fromarray(spt_masked_j))
            spt_masked.append(spt_masked_j)

            spt_lbls_orig_j = self.map_labels(spt_lbls_orig_j, int(pair[-1]))
            spt_lbls_orig_j = Image.fromarray(spt_lbls_orig_j)
            spt_lbls_orig.append(spt_lbls_orig_j)

            spt_trasnsformed = self.transform_sample({'image': spt_imgs_orig_j, 'label': spt_lbls_orig_j})
            spt_imgs_j = spt_trasnsformed['image']
            spt_lbls_j = spt_trasnsformed['label']
            spt_imgs.append(spt_imgs_j)

            spt_feats_file_j = self.root+pair[0][j].replace('JPEGImages', self.prefix_feat).replace('jpg', 'pt')
            try:
                spt_feats_j = torch.load(spt_feats_file_j)
            except FileNotFoundError:
                spt_feats_j = self.deeplab_model.module.extract_pixel_feature(spt_imgs_j[None, :].to('cuda')).detach().cpu()
                Path(spt_feats_file_j).parent.mkdir(parents=True, exist_ok=True)
                torch.save(spt_feats_j, spt_feats_file_j)

            spt_lbls_j = F.interpolate(spt_lbls_j[None, None, :], size=spt_feats_j.shape[-1])
            spt_feats_j = spt_feats_j.reshape(spt_feats_j.shape[1], -1)
            spt_lbls_j = spt_lbls_j.reshape(-1)
            spt_feats.append(spt_feats_j)
            spt_lbls.append(spt_lbls_j)

        return spt_imgs, spt_feats, spt_lbls, spt_masked, qry_img, qry_feat, qry_lbl, spt_imgs_orig, spt_lbls_orig, qry_img_orig, qry_lbl_orig, int(pair[-1])
        # return spt_imgs, spt_lbls, qry_img, qry_lbl, spt_imgs_orig, qry_img_orig, int(pair[-1])

    # def __getitem__(self, index):
    #     pair = self.oslsm_files[index]
    #     #self.out = self.PLP.load_next_frame(try_mode=False)
    #     original_im1 = []
    #     im1 = []
    #     lbl1= []

    #     #original_im2 = self.out['second_img'][0]
    #     original_im2 = cv2.imread(self.root+pair[1])
    #     original_im2 = cv2.resize(original_im2, self.img_size)

    #     im2 = np.asarray(cv2.imread(self.root+pair[1])[:,:,::-1], dtype=np.float32)
    #     lbl2 = cv2.imread(self.root+pair[1].replace('JPEGImages', self.prefix_lbl).replace('jpg', 'png') , 0)
    #     lbl2 = np.asarray(lbl2, dtype=np.int32)
    #     lbl2 = self.map_labels(lbl2, int(pair[-1]))

    #     im2, lbl2 = self.transform(im2, lbl2)

    #     for j in range(len(pair[0])):
    #         img = cv2.imread(self.root+pair[0][j])
    #         img = cv2.resize(img, self.img_size)
    #         original_im1.append(img)
    #         im1.append(np.asarray(cv2.imread(self.root+pair[0][j])[:,:,::-1], dtype=np.float32))
    #         temp_lbl = cv2.imread(self.root+pair[0][j].replace('JPEGImages', self.prefix_lbl).replace('jpg', 'png') , 0)
    #         temp_lbl = self.map_labels(temp_lbl, int(pair[-1]))
    #         lbl1.append(np.asarray(temp_lbl, dtype=np.int32))

    #         if self.is_transform:
    #             im1[j], lbl1[j] = self.transform(im1[j], lbl1[j])
    #     return im1, lbl1, im2, lbl2, original_im1, original_im2, int(pair[-1])#self.out['cls_ind']

    def correct_im(self, im):
        im = (np.transpose(im, (0,2,3,1)))/255.
        im += np.array([0.40787055,  0.45752459,  0.4810938])
        return im[:,:,:,::-1]
    # returns the outputs as images and also the first label in original img size

    def get_items_im(self):
        self.out = self.PLP.load_next_frame(try_mode=False)
        return (self.correct_im(self.out['first_img']),
                self.out['original_first_label'],
                self.correct_im(self.out['second_img']),
                self.out['second_label'][0],
                self.out['deploy_info'])

    def transform_sample(self, sample):
        composed_transforms = transforms.Compose([
            # tr_sample.FixScaleCrop(crop_size=self.args.crop_size),
            tr_sample.FixedResize(size=self.args.base_size),
            tr_sample.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr_sample.ToTensor()])

        return composed_transforms(sample)
    
    def transform_image(self, image):
        composed_transforms = transforms.Compose([
            # tr_image.FixScaleCrop(crop_size=self.args.crop_size),
            tr_image.FixedResize(size=self.args.base_size),
            tr_image.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr_image.ToTensor()])

        return composed_transforms(image)