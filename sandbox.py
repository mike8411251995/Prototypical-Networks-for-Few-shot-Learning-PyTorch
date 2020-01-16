import os, ntpath
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

from modeling.deeplab import *
from modeling.sync_batchnorm.replicate import patch_replication_callback
from dataloaders import *
from dataloaders import custom_transforms_image as tr_image
from dataloaders.utils import decode_seg_map_sequence

from ptsemseg.loader import pascalVOC5iLoader
from tqdm import tqdm

from Namespace import Namespace

args = Namespace(backbone='resnet', base_size=513, batch_size=5, checkname='deeplab-resnet', crop_size=513, cuda=True, dataset='pascal', epochs=50, eval_interval=1, freeze_bn=False, ft=False, gpu_ids=[0], loss_type='ce', lr=0.007, lr_scheduler='poly', momentum=0.9, nesterov=False, no_cuda=False, no_val=False, out_stride=16, resume='run/pascal/deeplab-resnet-all/model_best.pth.tar', seed=1, start_epoch=0, sync_bn=False, test_batch_size=5, use_balanced_weights=False, use_sbd=True, weight_decay=0.0005, workers=4)

model = DeepLab(num_classes=21,
                backbone=args.backbone,
                output_stride=args.out_stride,
                sync_bn=args.sync_bn,
                freeze_bn=args.freeze_bn)

# Using cuda
if args.cuda:
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    patch_replication_callback(model)
    model = model.cuda()

def transform_image(image):
    composed_transforms = transforms.Compose([
        tr_image.FixScaleCrop(crop_size=args.crop_size),
        tr_image.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr_image.ToTensor()])
    
    return composed_transforms(image)

def transform_sample(sample):
    composed_transforms = transforms.Compose([
        tr_sample.FixScaleCrop(crop_size=args.crop_size),
        tr_sample.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr_sample.ToTensor()])

    return composed_transforms(sample)

def inference(splt, img_path, label_path=None):
    args.resume = 'run/pascal/deeplab-resnet-split-{}/model_best.pth.tar'.format(splt)

    # Resuming checkpoint
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        if args.cuda:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))

    model.eval()

    # Inference
    image = Image.open(img_path).convert('RGB')
    if label_path:
        if label_path[-4:] == '.png':
            label = Image.open(label_path)
        elif label_path[-4:] == '.mat':
            label = Image.fromarray(scipy.io.loadmat(label_path)["GTcls"][0]['Segmentation'][0])
        sample = {'image': image, 'label': label}
        sample = transform_sample(sample)
        image = sample['image'][None, :, :]
        label = sample['label'][None, :, :]
    else:
        image = transform_image(image)
        image = image[None, :, :]

    if args.cuda:
        image = image.cuda()
    with torch.no_grad():
        output = model(image)

    img_name = ntpath.basename(img_path)[:-4]  + '_{}.jpg'.format(str(splt))
    num_of_subplots = 2

    pred = decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(), dataset=args.dataset)

    grid_image1 = make_grid(image.clone().cpu().data, 1, normalize=True)
    grid_image2 = make_grid(pred, 1, normalize=False, range=(0, 255))
        
    if label_path:
        target = decode_seg_map_sequence(label.detach().cpu().numpy(), dataset=args.dataset)
        grid_image3 = make_grid(target, 1, normalize=False, range=(0, 255)) 
        num_of_subplots = 3
    
    plt.figure()
    plt.subplot(1, num_of_subplots, 1)
    plt.imshow(grid_image1.permute(1, 2, 0))
    plt.subplot(1, num_of_subplots, 2)
    plt.imshow(grid_image2.permute(1, 2, 0))
    if label_path:
        plt.subplot(1, num_of_subplots, 3)
        plt.imshow(grid_image3.permute(1, 2, 0))

    plt.savefig(os.path.join('test_outputs', img_name))
    
# for i in [0, 1, 2, 3]:
#     img_path = 'datasets/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000241.jpg'
#     inference(i, img_path)

args.resume = 'run/pascal/deeplab-resnet-split-1/model_best.pth.tar'

# Resuming checkpoint
if args.resume is not None:
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
    checkpoint = torch.load(args.resume)
    if args.cuda:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))

model.eval()

data = pascalVOC5iLoader('datasets/VOC/VOCdevkit/VOC2012', args, model, inverse=True, fold=0, k_shot=1)
for i in tqdm(range(3000)):
    b = np.asarray(data[i][1][0])
    # print(data[i][1][0])