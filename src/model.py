import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from os.path import join as pjoin


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    def __init__(self, features=16):
        super(BasicBlock, self).__init__()
        self.Sp = nn.Softplus()

        self.conv_in = nn.Conv2d(1, features, (3, 3), stride=1, padding=1)
        self.conv1_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)

        self.conv1_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv_out = nn.Conv2d(features, 1, (3, 3), stride=1, padding=1)

    def forward(self, Z, PsiTPsi, PsiTYn, mu, theta):
        # gradient descent update, Eq. (7a)
        R = Z.squeeze(1) - self.Sp(mu) * (torch.bmm(PsiTPsi.squeeze(1), Z.squeeze(1)) - PsiTYn.squeeze(1))
        R = torch.unsqueeze(R, 1)

        # proximal mapping module, T transformation in Eq. (7b)
        R0 = R

        Rin = self.conv_in(R0.float())

        # F transformation
        R = self.conv1_forward(Rin)
        R = F.relu(R)
        R = self.conv2_forward(R)
        R = F.relu(R)
        R = self.conv3_forward(R)
        R = F.relu(R)
        Rforw = self.conv4_forward(R)

        # soft-thresholding block - save for sparsity loss LFspa
        Rst = torch.mul(torch.sign(Rforw), F.relu(torch.abs(Rforw) - self.Sp(theta)))

        # Ftilde transformation
        R = self.conv1_backward(Rst)
        R = F.relu(R)
        R = self.conv2_backward(R)
        R = F.relu(R)
        R = self.conv3_backward(R)
        R = F.relu(R)
        Rbackw = self.conv4_backward(R)

        Rout = self.conv_out(Rbackw)

        # prediction output (skip connection)
        Xnew = R0 + Rout

        # compute symmetry loss LFsym
        R = self.conv1_backward(Rforw)
        R = F.relu(R)
        R = self.conv2_backward(R)
        R = F.relu(R)
        R = self.conv3_backward(R)
        R = F.relu(R)
        Rin_est = self.conv4_backward(R)
        Rdiff = Rin_est - Rin

        return Xnew, Rdiff, Rst

    
class FISTANet(nn.Module):
    def __init__(self, layer_no, feature_no):
        """
        layer_no   : number of FISTA iteration layers
        feature_no : number of convolutional filters in F
        """
        super(FISTANet, self).__init__()

        # itaration layers
        self.layer_no = layer_no
        layer_list = []
        self.bb = BasicBlock(features=feature_no)
        for i in range(self.layer_no):
            layer_list.append(self.bb)
        self.fcs = nn.ModuleList(layer_list)
        self.fcs.apply(initialize_weights)

        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))

        self.Sp = nn.Softplus()

    def forward(self, X0, Yn, Psi):
        """
        X0  : initialized coefficient vector [batch_size x K x 1]
        Yn  : measured signal vector         [batch_size x n x 1]
        Psi : dictionary matrix              [batch_size x n x K]
        """
        # preprocess inputs
        PsiTPsi = torch.bmm(Psi.permute(0, 2, 1), Psi).unsqueeze(1)
        PsiTYn = torch.bmm(Psi.permute(0, 2, 1), Yn).unsqueeze(1)
        X0 = torch.unsqueeze(X0, 1)

        # initialize loop
        Xold = X0
        Z = Xold
        layers_Rdiff = []     # for computing symmetry loss LFsym
        layers_Rst = []       # for computing sparsity loss LFspa
    
        # FISTA-Net main loop of iteration layers
        for i in range(self.layer_no):
            theta_ = self.w_theta * i + self.b_theta                                 # Eq. (10a)
            mu_ = self.w_mu * i + self.b_mu                                          # Eq. (10b)
            [Xnew, Rdiff, Rst] = self.fcs[i](Z, PsiTPsi, PsiTYn, mu_, theta_)        # Eq. (7a), (7b)     
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) - self.Sp(self.b_rho)) /\
                        self.Sp(self.w_rho * i + self.b_rho)                         # Eq. (10c)
            Z = Xnew + rho_ * (Xnew - Xold)                                          # Eq. (7c)
            layers_Rdiff.append(Rdiff)
            layers_Rst.append(Rst)
  
        return Xnew.squeeze(1), layers_Rdiff, layers_Rst
