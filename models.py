# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:30:30 2019
@author: tbmoon, https://github.com/tbmoon/facenet

@Modifide by Haoxiong Ma
"""


import torch
import torch.nn as nn
from torchvision.models import resnet34
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance

class FaceNetModel(nn.Module):
    
    def __init__(self, embedding_size, num_classes, pretrained=False):
        super (FaceNetModel, self).__init__()
        
        self.model            = resnet34(pretrained)
        self.embedding_size   = embedding_size
        self.model.fc         = nn.Linear(12800, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)
        
    def l2_norm(self, input):
        input_size = input.size()
        buffer     = torch.pow(input, 2)
        normp      = torch.sum(buffer, 1).add_(1e-10)
        norm       = torch.sqrt(normp)
        _output    = torch.div(input, norm.view(-1, 1).expand_as(input))
        output     = _output.view(input_size)
    
        return output  
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        self.features = self.features*alpha
        
        return self.features   
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res      = self.model.classifier(features)       
        return res

class TripletLoss(Function):
    
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist  = PairwiseDistance(2)
        
    def forward(self, anchor, positive, negative):
        pos_dist   = self.pdist.forward(anchor, positive)
        neg_dist   = self.pdist.forward(anchor, negative)
        
        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min = 0.0)
        loss       = torch.mean(hinge_dist)
        return loss
