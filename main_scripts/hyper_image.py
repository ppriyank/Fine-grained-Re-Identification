from __future__ import print_function, absolute_import
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

storage_dir = "/beegfs/pp1953/"
import argparse
import configparser
import random 
import os.path as osp
import numpy as np


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
# from tools import * 
import models

from loss import CrossEntropyLabelSmooth, TripletLoss , CenterLoss , OSM_CAA_Loss , Satisfied_Rank_loss2
from tools.transforms2 import *
from tools.scheduler import WarmupMultiStepLR
from tools.utils import AverageMeter, Logger, save_checkpoint
from tools.eval_metrics import evaluate , re_ranking
from tools.samplers import RandomIdentitySampler
from tools.video_loader import ImageDataset , Image_inderase
import tools.data_manager as data_manager


from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
import ax
from typing import Dict, List, Tuple


print("Current File Name : ",os.path.realpath(__file__))



parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max-epoch', default=201, type=int,
                    help="maximum epochs to run")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=256, type=int)
parser.add_argument('--num-instances', type=int, default=4, help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default="ResNet50ta_bt5", help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
parser.add_argument('-n', '--mode-name', type=str, default='', help="ResNet50ta_bt2_supervised_erase_59_checkpoint_ep81.pth.tar, \
    ResNet50ta_bt2_supervised_erase_44_checkpoint_ep101.pth.tar")

# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--name', '--model_name', type=str, default='_supervised_erase_')
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-opt', '--opt', type=str, default='3', help="choose opt")
parser.add_argument('-s', '--sampling', type=str, default='random', help="choose sampling for training")
parser.add_argument('--thresold', type=int, default='60')
parser.add_argument('-f', '--focus', type=str, default='map', help="map,rerank_map")
parser.add_argument('--heads', default=1, type=int, help="no of heads of multi head attention")
parser.add_argument('--fin-dim', default=2048, type=int, help="final dim for center loss")
parser.add_argument('--rerank', default=False, type=bool)


args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("train_batch===" ,  args.train_batch , "seq_len" , args.seq_len, "no of gpus : " , os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count() )
else:
    print("train_batch===" ,  args.train_batch , "seq_len" , args.seq_len)

args.gpu_devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
cudnn.benchmark = True
pin_memory = True if use_gpu else False
path =  storage_dir + "mars_sota.pth.tar"
ranks=[1, 5, 10, 20]
def train(parameters: Dict[str, float]) -> nn.Module:
    global args 
    print("====", args.focus,  "=====")
    dataset = data_manager.init_dataset(name=args.dataset)
    transform_train = transforms.Compose([
                transforms.Resize((args.height, args.width), interpolation=3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Pad(10),
                Random2DTranslation(args.height, args.width),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    transform_test = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainloader = DataLoader(
        Image_inderase(dataset.train, seq_len=args.seq_len, sample=args.sampling,transform=transform_train),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )
    queryloader = DataLoader(
        ImageDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test,eval=True),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )
    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test,eval=True),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    lamb = 0.3
    base_learning_rate = 0.00035

    margin = parameters.get("margin", None)
    alpha = parameters.get("alpha", None)
    l = parameters.get("l", None)
    sigma = parameters.get("sigma", None)
    weight_decay = parameters.get("weight_decay", None)
    beta_ratio = parameters.get("beta_ratio", None)
    gamma = parameters.get("gamma", None)

    lsr_weight = parameters.get("lsr_weight", None)
    lcn_weight = parameters.get("lcn_weight", None)
    H2_weight = parameters.get("H2_weight", None)
    var_weight = parameters.get("var_weight", None)
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids , fin_dim=args.fin_dim, heads=4)
    
    print( "Loading MARS model" )
    if use_gpu:
        checkpoint = torch.load( path  )
    else:
        checkpoint = torch.load( path , map_location=torch.device('cpu') )
    state_dict = {}
    state_dict2 = {}
    for key in checkpoint['state_dict']:
        if "base" in  key :
            temp = key.replace("base." , "")
            state_dict[temp] = checkpoint['state_dict'][key]

    model.base_mars.load_state_dict(state_dict,  strict=True)
    del  state_dict , state_dict2, checkpoint

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_htri = TripletLoss(margin, 'cosine')
    criterion_xent = CrossEntropyLabelSmooth(dataset.num_train_pids,use_gpu=use_gpu)
    criterion_center_loss = CenterLoss(use_gpu=use_gpu , feat_dim=args.fin_dim  )
    criterion_osm_caa = OSM_CAA_Loss(alpha=alpha , l=l , osm_sigma=sigma ,use_gpu=use_gpu)
    criterion_lsr = Satisfied_Rank_loss2(use_gpu=use_gpu)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_learning_rate
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    optimizer = torch.optim.Adam(params)
    scheduler = WarmupMultiStepLR(optimizer, milestones=[40, 70], gamma=gamma, warmup_factor=0.01, warmup_iters=10)
    optimizer_center = torch.optim.SGD(criterion_center_loss.parameters(), lr=0.5)
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    num_epochs = 121
    for epoch in range (num_epochs):
        val = train_model(model, beta_ratio, trainloader, var_weight, criterion_xent, criterion_htri, optimizer, optimizer_center , criterion_center_loss,  \
                    lsr_weight , lcn_weight , H2_weight, criterion_lsr , criterion_osm_caa)
        if math.isnan(val[-2]):
            return 0
        scheduler.step()
        if epoch % 40 ==0 :
            print("Var ({:.6f}) H2 ({:.6f}) lcns ({:.6f}) l_sr ({:.6f})  TripletLoss  ({:.6f}) OSM Loss: ({:.6f}) Total Loss {:.6f} ({:.6f})  ".format(val[0], val[1], val[2] , val[3] ,val[4], val[5],val[6], val[7]))        
            
    result1= test_rerank(model, queryloader, galleryloader, lamb=lamb, parameters=parameters )    
    del queryloader
    del galleryloader
    del trainloader 
    del model
    del criterion_htri
    del criterion_xent
    del criterion_center_loss
    del criterion_osm_caa
    del optimizer
    del optimizer_center
    del scheduler
    return result1



def train_model(model, beta_ratio, trainloader, var_weight, criterion_xent, criterion_htri, optimizer, optimizer_center , criterion_center_loss , lsr_weight , lcn_weight , 
    H2_weight , criterion_lsr, criterion_osm_caa):
    model.train()
    losses = AverageMeter()
    cetner_loss_weight = 0.0005
    for batch_idx, (imgs, pids) in enumerate(trainloader):
        pids = pids.view(-1)
        imgs  = imgs.view(-1, 3, args.height, args.width)

        if use_gpu:
            imgs, pids   = imgs.cuda(), pids.cuda() 
        # labels =labels.float()
        imgs, pids = Variable(imgs), Variable(pids)
        # print(labels)
        y1,y2 , features , H2  =  model(imgs)
        if use_gpu:
            targets = torch.zeros(y1.size()).scatter_(1, pids.unsqueeze(1).data.cpu(), 1).cuda()
        else:
            targets = torch.zeros(y1.size()).scatter_(1, pids.unsqueeze(1).data.cpu(), 1)
        proto = torch.mm(targets.t() , features)
        proto = proto / 4
        proto = proto[pids]
        var = (features - proto).pow(2).sum(0) /  ( (args.seq_len - 1) *  args.train_batch // args.seq_len)
        var = var_weight * var.sum()

        ide_loss1 = criterion_xent(y1 , pids)
        ide_loss2 = criterion_xent(y2 , pids)
        # regulaization paper
        ide_loss = (ide_loss1 + ide_loss2) / 2
        triplet_loss = criterion_htri(features, features, features, pids, pids, pids)
        center_loss = criterion_center_loss(features, pids)
        osm_caa_loss = criterion_osm_caa(features, pids , criterion_center_loss.centers.t() ) 
        lsr , lcns =  criterion_lsr(y1= y1, y2=y2, label = pids, T=None)
        
        loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss + var  + lsr_weight * lsr + lcn_weight * lcns + H2_weight *  H2 

        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer.step()
        for param in criterion_center_loss.parameters():
            param.grad.data *= (1./cetner_loss_weight)
        optimizer_center.step()
        losses.update(loss.data.item(), pids.size(0))
        # print(model.module.gamma)
    return var.item(), H2.item(), lcns.item() , lsr.item() ,triplet_loss.item(), osm_caa_loss.item(),losses.val, losses.avg

def test_rerank(model, queryloader, galleryloader, lamb=None , parameters=None):
    model.eval()
    global temp_count
    qf, q_pids, q_camids = [], [], []
        
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            imgs  = imgs.view(-1, 3, args.height, args.width)
            features = model(imgs)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            imgs  = imgs.view(-1, 3, args.height, args.width)
            features = model(imgs)  
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()
        gf = gf.numpy()
        qf = qf.numpy()
        distmat_rerank = None
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Results ---------- ")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1] ))
        print("------------------")
        print(parameters)
        return mAP



args.save_dir = storage_dir + "resnet/"
args.name += "_" + args.dataset + "_" + str(args.heads) + "_"
is_best = 0
prev_best = 0 


if args.mode_name != '':
    name = args.mode_name
    # name = "ResNet50ta_bt2_supervised_erase_59_checkpoint_ep81.pth.tar"
    print("loading .... " , name)
    if use_gpu:
        checkpoint = torch.load(osp.join(args.save_dir, name)  )
    else:
        checkpoint = torch.load(osp.join(args.save_dir, name) , map_location=torch.device('cpu') )
    state_dict = {}
    if args.dataset != "mars":
        for key in checkpoint['state_dict']:
            if "classifier" not in  key:
                state_dict["module." + key] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict,  strict=False)
    else:
        for key in checkpoint['state_dict']:
            state_dict["module." + key] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict,  strict=True)



best_parameters, values, experiment, model = optimize(
    parameters=[
    {"name": "margin", "type": "range", "bounds": [1e-6, 1.0], "log_scale": True},
    {"name": "alpha", "type": "range", "bounds": [0.5, 3.0]},
    {"name": "l", "type": "range", "bounds": [1e-1, 1.0]},
    {"name": "sigma", "type": "range", "bounds": [1e-1, 1.0]},
    {"name": "weight_decay", "type": "range", "bounds": [1e-6, 1.0]},
    {"name": "gamma", "type": "range", "bounds": [1e-6, 1.0]},
    {"name": "beta_ratio", "type": "range", "bounds": [1e-6, 1.0]},
    {"name": "var_weight", "type": "range", "bounds": [1e-6, 1e-2]},
    {"name": "H2_weight", "type": "range", "bounds": [1e-2, 10.0]},
    {"name": "lsr_weight", "type": "range", "bounds": [1e-2, 3.0]},
    {"name": "lcn_weight", "type": "range", "bounds": [1e-2, 3.0]},
    ],
    evaluation_function=train,
    objective_name='ranking',
    minimize=False,
    total_trials = 120,
)

print("===========")
print(best_parameters)
print("===========")
print(values)




# python hyoer_image.py -d=market --opt=24 --thresold=0 --max-epoch=500 -a="ResNet50ta_bt10" --sampling="intelligent" 
# python hyper_image.py -d=market_sub -a="ResNet50ta_bt10" --sampling="intelligent" 
