from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from tools.helplayer import weights_init_kaiming 

__all__ = [ 'ResNet50TA_BT_image', 'ResNet50TA_BT_video']
 
#num classes = 625
#input x = (32, 4, 3, 224, 112)
#person_ids ids =  (32,)
#cam_id c = (32,)

class ResNet50TA_BT_video(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50TA_BT_video, self).__init__()
        self.base = torchvision.models.resnet50(pretrained=True)
    
        self.base.layer4[0].conv2.stride = (1,1)
        self.base.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(*list(self.base.children())[:-2])

        self.base_mars = torchvision.models.resnet50()
        self.base_mars.layer4[0].conv2.stride = (1,1)
        self.base_mars.layer4[0].downsample[0].stride = (1,1)
        self.base_mars = nn.Sequential(*list(self.base_mars.children())[:-2])
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.att_gen = 'softmax'
        self.final_dim = kwargs["fin_dim"]

        self.num_classes = num_classes
        self.feat_dim = 2048 # feature dimension
        self.middle_dim1 = 256 # middle layer dimension
        self.middle_dim2 = 1024 # middle layer dimension
        
        self.conv1 = nn.Conv2d(self.feat_dim, self.middle_dim2, 1)
        self.conv2 = nn.Conv2d(self.feat_dim, self.middle_dim2, 1)

        self.bn1 = nn.BatchNorm1d(self.middle_dim2)
        self.bn2 = nn.BatchNorm1d(self.middle_dim2)

        self.classifier = nn.Linear(self.middle_dim2 , self.num_classes, bias=False)

        self.theta = nn.Conv2d(self.middle_dim2 , self.middle_dim1 , kernel_size=[1,1], bias=False)
        self.g = nn.Conv2d(self.middle_dim2 , self.middle_dim1 , kernel_size=[1,1], bias=False )
        self.w_z = nn.Conv2d(self.middle_dim1 , self.middle_dim2 , kernel_size=1, bias=False)

        self.conv2.apply(weights_init_kaiming)
        self.conv1.apply(weights_init_kaiming)
        self.w_z.weight.data.fill_(0.0)
        
        
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        mars_x = self.base_mars(x)
        x = self.base(x)

        h_temp = x.size(2)
        w_temp =  x.size(3)

        x = self.conv1(x)
        A_gap = self.gap(x)

        A_gap = A_gap.view(b*t,self.middle_dim2)
        scores = A_gap.softmax(1)
        
        scores=  scores.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        attention_maps = (scores * (x - torch.min(x)) ).sum(1)
        attention_maps = torch.sigmoid(attention_maps.unsqueeze(1))
        
        A_gap = A_gap.view(b,t, self.middle_dim2)
        F1 = A_gap.mean(1)
        F1 = self.bn1(F1)
        y1 = self.classifier(F1)

        mars_x = self.conv2(mars_x)
        map = attention_maps
        AiF1 = mars_x * attention_maps

        AiF1 = AiF1.view(b,t,self.middle_dim2, -1)
        AiF1 = AiF1.permute(0,2,1,3)

        f1 = F.relu(self.theta(AiF1))
        f1 = f1.reshape(b,self.middle_dim1, -1)
        f1 = F.normalize(f1, p=2, dim=1)
        f1 = torch.bmm(f1.permute(0,2,1) , f1)
        f1 = F.softmax(f1, dim=-1)

        f2 = F.relu(self.g(AiF1))
        f2 = f2.reshape(b,self.middle_dim1, -1)
        f2 = f2.permute(0,2,1)
        f2 = torch.bmm(f1 , f2)
        f2 = f2.permute( 0, 2,1).reshape(b,self.middle_dim1, t , -1)
        
        AiF1 = self.w_z(f2) + AiF1            
        AiF1 = AiF1.sum(-1) / map.view(b,1,t,-1).expand_as(AiF1).sum(-1)
        AiF1 = AiF1.mean(-1).unsqueeze(-1)
        
        AiF1=  AiF1.squeeze(-1)
        F2 = self.bn2(AiF1)
        y2 = self.classifier(F2)

        x = torch.cat([F1, F2], 1)
        if not self.training:
            return x
        return y1,y2 , x 



class ResNet50TA_BT_image(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50TA_BT_image, self).__init__()
        self.base = torchvision.models.resnet50(pretrained=True)
        
        self.base.layer4[0].conv2.stride = (1,1)
        self.base.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(*list(self.base.children())[:-2])

        self.base_mars = torchvision.models.resnet50()
        self.base_mars.layer4[0].conv2.stride = (1,1)
        self.base_mars.layer4[0].downsample[0].stride = (1,1)
        self.base_mars = nn.Sequential(*list(self.base_mars.children())[:-2])
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.att_gen = 'softmax'
        self.final_dim = kwargs["fin_dim"]

        self.num_classes = num_classes
        self.feat_dim = 2048 # feature dimension
        self.middle_dim1 = 256 # middle layer dimension
        self.middle_dim2 = 1024 # middle layer dimension
        
        self.conv1 = nn.Conv2d(self.feat_dim, self.middle_dim2, 1)
        self.conv2 = nn.Conv2d(self.feat_dim, self.middle_dim2, 1)

        self.bn1 = nn.BatchNorm1d(self.middle_dim2)
        self.bn2 = nn.BatchNorm1d(self.middle_dim2)

        self.classifier = nn.Linear(self.middle_dim2 , self.num_classes, bias=False)

        self.theta = nn.Conv2d(self.middle_dim2 , self.middle_dim1 , kernel_size=[1,1], bias=False)
        self.g = nn.Conv2d(self.middle_dim2 , self.middle_dim1 , kernel_size=[1,1], bias=False )
        self.w_z = nn.Conv2d(self.middle_dim1 , self.middle_dim2 , kernel_size=1, bias=False)

        self.conv2.apply(weights_init_kaiming)
        self.conv1.apply(weights_init_kaiming)
        self.w_z.weight.data.fill_(0.0)
        
    def forward(self, x):
        
        b = x.size(0)
        mars_x = self.base_mars(x)
        x = self.base(x)

        h_temp = x.size(2)
        w_temp =  x.size(3)

        x = self.conv1(x)
        A_gap = self.gap(x)
        A_gap = A_gap.view(b,self.middle_dim2)
        scores = A_gap.softmax(1)
        
        scores=  scores.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        attention_maps = (scores * (x - torch.min(x)) ).sum(1)
        attention_maps = torch.sigmoid(attention_maps.unsqueeze(1))
        
        A_gap = A_gap.view(b, self.middle_dim2)
        F1 = A_gap
        F1 = self.bn1(F1)
        y1 = self.classifier(F1)

        mars_x = self.conv2(mars_x)    
        map = attention_maps
        AiF1 = mars_x * attention_maps
        
        f1 = F.relu(self.theta(AiF1))
        f1 = f1.reshape(b,self.middle_dim1, -1)
        f1 = F.normalize(f1, p=2, dim=1)
        f1 = torch.bmm(f1.permute(0,2,1) , f1)
        f1 = F.softmax(f1, dim=-1)

        f2 = F.relu(self.g(AiF1))
        f2 = f2.reshape(b,self.middle_dim1, -1)
        f2 = f2.permute(0,2,1)
        f2 = torch.bmm(f1 , f2)
        f2 = f2.permute( 0, 2,1).reshape(b,self.middle_dim1, h_temp, w_temp )
        
        AiF1 = self.w_z(f2) + AiF1            
        AiF1 = AiF1.sum(-1).sum(-1) / map.view(b,1,-1).sum(-1).expand(b, self.middle_dim2)
        AiF1 = AiF1.unsqueeze(-1)
        AiF1=  AiF1.squeeze(-1)
        F2 = self.bn2(AiF1)
        y2 = self.classifier(F2)

        x = torch.cat([F1, F2], 1)
        if not self.training:
            return x
        return y1,y2 , x 


