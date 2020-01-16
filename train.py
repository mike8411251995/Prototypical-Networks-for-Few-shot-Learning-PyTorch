# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from parser_util import get_parser

from tqdm import tqdm
import numpy as np
import torch
import os

from Namespace import Namespace
from modeling.deeplab import *
from modeling.sync_batchnorm.replicate import patch_replication_callback
from ptsemseg.loader import pascalVOC5iLoader

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc_foreground = []
    train_acc_background = []
    val_loss = []
    val_acc_foreground = []
    val_acc_background = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        model.train()
        for data in tqdm(tr_dataloader):
            optim.zero_grad()
            _, spt_feats, spt_lbls, _, _, qry_feat, qry_lbl, _, _, _, _, class_id = data

            ############################################################################################

            spt_foreground_idx = torch.nonzero((spt_lbls[0] == class_id).view(-1).data).squeeze()
            if spt_foreground_idx.size(0) >= 300:
                perm = torch.randperm(spt_foreground_idx.size(0))
                idx = perm[:300]
                spt_foreground_pix_samples = spt_feats[0][:, spt_foreground_idx[idx]]
                spt_foreground_pix_labels = spt_lbls[0][spt_foreground_idx[idx]]
            else:
                continue

            spt_background_idx = torch.nonzero((spt_lbls[0] == 0).view(-1).data).squeeze()
            if spt_background_idx.size(0) >= 300:
                perm = torch.randperm(spt_background_idx.size(0))
                idx = perm[:300]
                spt_background_pix_samples = spt_feats[0][:, spt_background_idx[idx]]
                spt_background_pix_labels = spt_lbls[0][spt_background_idx[idx]]
            else:
                continue

            ############################################################################################

            qry_foreground_idx = torch.nonzero((qry_lbl == class_id).view(-1).data).squeeze()
            qry_foreground_pix_samples = qry_feat[:, qry_foreground_idx]
            qry_foreground_pix_labels = qry_lbl[qry_foreground_idx]

            qry_background_idx = torch.nonzero((qry_lbl == 0).view(-1).data).squeeze()
            qry_background_pix_samples = qry_feat[:, qry_background_idx]
            qry_background_pix_labels = qry_lbl[qry_background_idx]

            ############################################################################################

            perm = torch.randperm(600)
            x_spt = torch.cat((spt_foreground_pix_samples, spt_background_pix_samples), 1).permute(1, 0)[perm][None, :].permute(1, 0, 2).to(device)
            y_spt = torch.cat((spt_foreground_pix_labels, spt_background_pix_labels), 0)[perm].to(device)
            spt_output = model(x_spt)

            x_qry = torch.cat((qry_foreground_pix_samples, qry_background_pix_samples), 1).permute(1, 0)[None, :].permute(1, 0, 2).to(device)
            y_qry = torch.cat((qry_foreground_pix_labels, qry_background_pix_labels), 0).to(device)
            qry_output = model(x_qry)

            loss, acc_foreground, acc_background = loss_fn(spt_output, y_spt, qry_output, y_qry, class_id)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc_foreground.append(acc_foreground.item())
            train_acc_background.append(acc_background.item())

        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc_foreground = np.mean(train_acc_foreground[-opt.iterations:])
        avg_acc_background = np.mean(train_acc_background[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc (Foreground): {}, Avg Train Acc (Background): {}'.format(avg_loss, avg_acc_foreground, avg_acc_background))
        lr_scheduler.step()

        if val_dataloader is None:
            continue
        model.eval()
        for data in tqdm(val_dataloader):
            _, spt_feats, spt_lbls, _, _, qry_feat, qry_lbl, _, _, _, _, class_id = data

            ############################################################################################

            spt_foreground_idx = torch.nonzero((spt_lbls[0] == class_id).view(-1).data).squeeze()
            if spt_foreground_idx.size(0) >= 300:
                perm = torch.randperm(spt_foreground_idx.size(0))
                idx = perm[:300]
                spt_foreground_pix_samples = spt_feats[0][:, spt_foreground_idx[idx]]
                spt_foreground_pix_labels = spt_lbls[0][spt_foreground_idx[idx]]
            else:
                continue

            spt_background_idx = torch.nonzero((spt_lbls[0] == 0).view(-1).data).squeeze()
            if spt_background_idx.size(0) >= 300:
                perm = torch.randperm(spt_background_idx.size(0))
                idx = perm[:300]
                spt_background_pix_samples = spt_feats[0][:, spt_background_idx[idx]]
                spt_background_pix_labels = spt_lbls[0][spt_background_idx[idx]]
            else:
                continue

            ############################################################################################

            qry_foreground_idx = torch.nonzero((qry_lbl == class_id).view(-1).data).squeeze()
            qry_foreground_pix_samples = qry_feat[:, qry_foreground_idx]
            qry_foreground_pix_labels = qry_lbl[qry_foreground_idx]

            qry_background_idx = torch.nonzero((qry_lbl == 0).view(-1).data).squeeze()
            qry_background_pix_samples = qry_feat[:, qry_background_idx]
            qry_background_pix_labels = qry_lbl[qry_background_idx]

            ############################################################################################

            perm = torch.randperm(600)
            x_spt = torch.cat((spt_foreground_pix_samples, spt_background_pix_samples), 1).permute(1, 0)[perm][None, :].permute(1, 0, 2).to(device)
            y_spt = torch.cat((spt_foreground_pix_labels, spt_background_pix_labels), 0)[perm].to(device)
            spt_output = model(x_spt)

            x_qry = torch.cat((qry_foreground_pix_samples, qry_background_pix_samples), 1).permute(1, 0)[None, :].permute(1, 0, 2).to(device)
            y_qry = torch.cat((qry_foreground_pix_labels, qry_background_pix_labels), 0).to(device)
            qry_output = model(x_qry)

            loss, acc_foreground, acc_background = loss_fn(spt_output, y_spt, qry_output, y_qry, class_id)
            val_loss.append(loss.item())
            val_acc_foreground.append(acc_foreground.item())
            val_acc_background.append(acc_background.item())

        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc_foreground = np.mean(val_acc_foreground[-opt.iterations:])
        avg_acc_background = np.mean(val_acc_background[-opt.iterations:])
        avg_acc = float((avg_acc_foreground + avg_acc_background) / 2)

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        print('Avg Val Loss: {}, Avg Val Acc (Foreground): {}, Avg Val Acc (Background): {}{}'.format(
            avg_loss, avg_acc_foreground, avg_acc_background, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
        
    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc_foreground', 'train_acc_background', 'val_loss', 'val_acc_foreground', 'val_acc_background']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc_foreground, train_acc_background, val_loss, val_acc_foreground, val_acc_background


# def test(opt, test_dataloader, model):
#     '''
#     Test the model trained with the prototypical learning algorithm
#     '''
#     device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
#     avg_acc = list()
#     for epoch in range(10):
#         test_iter = iter(test_dataloader)
#         for batch in test_iter:
#             x, y = batch
#             x, y = x.to(device), y.to(device)
#             model_output = model(x)
#             _, acc = loss_fn(model_output, target=y,
#                              n_support=opt.num_support_val)
#             avg_acc.append(acc.item())
#     avg_acc = np.mean(avg_acc)
#     print('Test Acc: {}'.format(avg_acc))

#     return avg_acc


# def eval(opt):
#     '''
#     Initialize everything and train
#     '''
#     options = get_parser().parse_args()

#     if torch.cuda.is_available() and not options.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#     init_seed(options)
#     test_dataloader = init_dataset(options)[-1]
#     model = init_protonet(options)
#     model_path = os.path.join(opt.experiment_root, 'best_model.pth')
#     model.load_state_dict(torch.load(model_path))

#     test(opt=options,
#          test_dataloader=test_dataloader,
#          model=model)


def main():
    '''
    Initialize everything and train
    '''
    fold = 0

    args = Namespace(backbone='resnet', base_size=513, crop_size=513, cuda=True, dataset='pascal', freeze_bn=False, gpu_ids=[0], out_stride=16, sync_bn=False)

    deeplab_model = DeepLab(num_classes=21,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    deeplab_model = torch.nn.DataParallel(deeplab_model, device_ids=args.gpu_ids)
    patch_replication_callback(deeplab_model)
    deeplab_model = deeplab_model.cuda()

    args.resume = 'run/pascal/deeplab-resnet-split-{}/model_best.pth.tar'.format(str(fold))
    checkpoint = torch.load(args.resume)
    deeplab_model.module.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    deeplab_model.eval()

    # options = get_parser().parse_args()
    options = Namespace(cuda=True, dataset_root='dataset', epochs=100, experiment_root='output', iterations=100, learning_rate=0.001, lr_scheduler_gamma=0.5, lr_scheduler_step=20, manual_seed=7)
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    
    tr_dataloader = pascalVOC5iLoader('datasets/VOC/VOCdevkit/VOC2012', args, deeplab_model, inverse=True, fold=fold, k_shot=1)
    val_dataloader = pascalVOC5iLoader('datasets/VOC/VOCdevkit/VOC2012', args, deeplab_model, inverse=False, fold=fold, k_shot=1)

    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc_foreground, train_acc_foreground, val_loss, val_acc_foreground, val_acc_background = res
    
    # print('Testing with last model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)

    # model.load_state_dict(best_state)
    # print('Testing with best model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)

    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)


if __name__ == '__main__':
    main()
