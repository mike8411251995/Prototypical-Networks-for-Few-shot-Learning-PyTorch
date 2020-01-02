import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_V2S(nn.Module):
    def __init__(self, opt):
        super(MLP_V2S, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.f_hid)
        self.fc2 = nn.Linear(opt.f_hid, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        # self.lrelu = nn.ReLU(True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        h = self.relu(self.fc2(h))
        # h = self.fc2(h)
        return h

class DomainClassifier(nn.Module):
    def __init__(self,dim):
        super(DomainClassifier, self).__init__()

        self.fcD = nn.Sequential(
            nn.Linear(dim, 2)
        )

        self.apply(weights_init)

    def forward(self, x):
        x = self.fcD(F.relu(x))
        return x