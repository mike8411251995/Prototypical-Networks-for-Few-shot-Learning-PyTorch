# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(spt_output, y_spt, qry_output, y_qry, class_id):
    spt_output_cpu = spt_output.to('cpu')
    y_spt_cpu = y_spt.to('cpu')
    qry_output_cpu = qry_output.to('cpu')
    y_qry_cpu = y_qry.to('cpu')

    def supp_idxs(c):
        return y_spt_cpu.eq(c).nonzero().squeeze(1)

    classes = torch.tensor([0., class_id])
    n_classes = len(classes)

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([spt_output_cpu[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs_foreground = y_qry_cpu.eq(class_id).nonzero().squeeze(1)
    query_idxs_background = y_qry_cpu.eq(0).nonzero().squeeze(1)

    query_samples_foreground = qry_output_cpu[query_idxs_foreground]
    query_samples_background = qry_output_cpu[query_idxs_background]
    dists_foreground = euclidean_dist(query_samples_foreground, prototypes)
    dists_background = euclidean_dist(query_samples_background, prototypes)

    log_p_y_foreground = F.log_softmax(-dists_foreground, dim=1)
    log_p_y_background = F.log_softmax(-dists_background, dim=1)

    loss_val = -log_p_y_foreground[:, 1].mean() + -log_p_y_background[:, 0].mean()

    _, y_hat_foreground = log_p_y_foreground.max(1)
    _, y_hat_background = log_p_y_background.max(1)
    acc_val_foreground = y_hat_foreground.eq(1).float().mean()
    acc_val_background = y_hat_background.eq(0).float().mean()

    return loss_val, acc_val_foreground, acc_val_background
