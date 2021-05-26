from __future__ import print_function, absolute_import
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

storage_dir = "/scratch/pp1953/"
# storage_dir = "/scratch/pp1953/"
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
from tools.video_loader import VideoDataset , VideoDataset_inderase
import tools.data_manager as data_manager


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
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
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
parser.add_argument('--heads', default=4, type=int, help="no of heads of multi head attention")
parser.add_argument('--fin-dim', default=2048, type=int, help="final dim for center loss")

parser.add_argument('--pretrain', action='store_true', help="evaluation only")


args = parser.parse_args()
use_gpu = torch.cuda.is_available()

# random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# np.random.seed(args.seed)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)


if use_gpu:
    print("train_batch===" ,  args.train_batch , "seq_len" , args.seq_len, "no of gpus : " , os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count() )
else:
    print("train_batch===" ,  args.train_batch , "seq_len" , args.seq_len)

args.gpu_devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

cudnn.benchmark = True


if args.dataset != "ilidsvid" and args.dataset != "prid":
    dataset = data_manager.init_dataset(name=args.dataset)
else:
    print("Split -- ", args.seed)
    dataset = data_manager.init_dataset(name=args.dataset, split_id=args.seed)





pin_memory = True if use_gpu else False
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
    VideoDataset_inderase(dataset.train, seq_len=args.seq_len, sample=args.sampling,transform=transform_train),
    sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
    batch_size=args.train_batch, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=True,
)

if args.dataset == "mars":
    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test, max_length=32),
        batch_size=args.test_batch, shuffle=False, num_workers=2,
        pin_memory=pin_memory, drop_last=False,
    )
    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test, max_length=32),
        batch_size=args.test_batch, shuffle=False, num_workers=2,
        pin_memory=pin_memory, drop_last=False,
    )
else:
    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )
    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

print(args)
opt = args.opt
# opt = "4"
lamb = 0.3

base_learning_rate = 0.00035
config = configparser.ConfigParser()

dirpath = os.getcwd() 

# config.read('/home/pp1953/code/Video-Person-ReID-master/tools/val.conf')    

if args.dataset == "mars":
    print("val.conf")
    config.read(dirpath + "/../tools/val.conf")        
elif args.dataset == "prid":
    print("val_prid.conf")
    config.read(dirpath + "/../tools/val_prid.conf")
elif args.dataset == "ilidsvid":
    print("val_ilidsvid.conf")
    config.read(dirpath + "/../tools/val_ilidsvid.conf")
else:
    print("val.conf")
    config.read(dirpath + "/../tools/val.conf")        

margin =  float(config[opt]['margin'])
alpha =  float(config[opt]['alpha'])
l = float(config[opt]['l'])
sigma = float(config[opt]['sigma'])
gamma  = float(config[opt]['gamma'])
beta_ratio = float(config[opt]['beta_ratio'])

if 'var_weight' in config[opt]:
    var_weight = float(config[opt]['var_weight'])
else:
    var_weight = 0.01 * 0.001

if 'lsr_weight' in config[opt]:
    lsr_weight = float(config[opt]['lsr_weight'])
else:
    lsr_weight = 1

if 'lcn_weight' in config[opt]:
    lcn_weight = float(config[opt]['lcn_weight'])
else:
    lcn_weight = 1


if 'weight_decay' in config[opt]:
    weight_decay = float(config[opt]['weight_decay'])
else:
    weight_decay = 0.0005

if 'batch_size' in config[opt]:
    batch_size = int(config[opt]['batch_size'])
else:
    batch_size = 32





attention_heads = None 
model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids , fin_dim=args.fin_dim, heads=args.heads)
# model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids )
# 
if args.pretrain :
    print("LOADING PRETRAINED MARS")
    if args.dataset == "mars":
        if args.name == '1':
            path =storage_dir +  "resnet/bt15_dukevideo_220_150_32_5_4_95_2_59.pth.tar"
        elif args.name == '2':
            path =storage_dir + "resnet/bt15_dukevideo_220_150_32_5_4_95_3_59.pth.tar"
        elif args.name == '3':
            path = storage_dir + "resnet/bt15_dukevideo_250_150_32_4_4_95_0_59.pth.tar"
        else:
            path = storage_dir + "resnet/bt15_dukevideo_250_150_32_4_4_95_0_59.pth.tar"
        print( "Loading ** NEW ** DUKE VIDEO  model" )
        if use_gpu:
            checkpoint = torch.load( path  )
        else:
            checkpoint = torch.load( path , map_location=torch.device('cpu') )
        state_dict = {}
        state_dict2 = {}
        for key in checkpoint['state_dict']:
            if "base_mars" in  key :
                temp = key.replace("base_mars." , "")
                state_dict[temp] = checkpoint['state_dict'][key]
        # path =  storage_dir + "resnet/mars_sota.pth.tar"
        # print( "Loading MARS model" )
        # if use_gpu:
        #     checkpoint = torch.load( path  )
        # else:
        #     checkpoint = torch.load( path , map_location=torch.device('cpu') )
        # state_dict = {}
        # state_dict2 = {}
        # for key in checkpoint['state_dict']:
        #     if "base" in  key :
        #         temp = key.replace("base." , "")
        #         state_dict[temp] = checkpoint['state_dict'][key]
    else:
        if True :
            path =  storage_dir + "dataset/mars_sota.pth.tar"
            print( "Loading ** OLD ** MARS model"  , path)
            
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
        else:
            path =  storage_dir + "dataset/bt15_mars_250_150_32_5_4_86_1_53.pth.tar"
            print( "Loading ** NEW ** MARS model"  , path)
            if use_gpu:
                checkpoint = torch.load( path  )
            else:
                checkpoint = torch.load( path , map_location=torch.device('cpu') )
            state_dict = {}
            state_dict2 = {}
            for key in checkpoint['state_dict']:
                if "base_mars" in  key :
                    temp = key.replace("base_mars." , "")
                    state_dict[temp] = checkpoint['state_dict'][key]

    # import pdb
    # pdb.set_trace()
    # print(model.attention_conv.weight[200][100])
    # print(model.base_mars[0].weight[40][0])
    model.base_mars.load_state_dict(state_dict,  strict=True)
    # print(model.attention_conv.weight[200][100])
    # model.attention_conv.weight
    # .load_state_dict(state_dict,  strict=True)
    # model.base_mars.state_dict().keys()
    del  state_dict , state_dict2, checkpoint

print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

criterion_htri = TripletLoss(margin, 'cosine')
criterion_xent = CrossEntropyLabelSmooth(dataset.num_train_pids,use_gpu=use_gpu)
criterion_center_loss = CenterLoss(use_gpu=use_gpu , feat_dim=args.fin_dim  , num_classes= dataset.num_train_pids)
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


def train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa=None,):
    model.train()
    losses = AverageMeter()
    cetner_loss_weight = 0.0005
    global beta_ratio  , var_weight  , lsr_weight , lcn_weight
    global temp_count
    H2 = 0 
    reg = 0 
    H2_weight = 0 
    ide_loss3 = 0 
    for batch_idx, (imgs, pids, _, labels) in enumerate(trainloader):
        if use_gpu:
            imgs, pids  , labels = imgs.cuda(), pids.cuda() , labels.cuda()
        # labels =labels.float()
        imgs, pids = Variable(imgs), Variable(pids)
        # print(labels)
        y1,y2 , features  =  model(imgs)

        # regulaization paper
        if use_gpu:
            targets = torch.zeros(y1.size()).scatter_(1, pids.unsqueeze(1).data.cpu(), 1).cuda()
        else:
            targets = torch.zeros(y1.size()).scatter_(1, pids.unsqueeze(1).data.cpu(), 1)

        proto = torch.mm(targets.t() , features)
        proto = proto / 4
        proto = proto[pids]
        var = (features - proto).pow(2).sum(0) /  ( (args.seq_len - 1) *  args.train_batch // args.seq_len)
        var = var_weight * var.sum()
        # regulaization paper

        ide_loss1 = criterion_xent(y1 , pids)
        ide_loss2 = criterion_xent(y2 , pids)
        
        ide_loss = (ide_loss1 + ide_loss2) / 2
        triplet_loss = criterion_htri(features, features, features, pids, pids, pids)
        center_loss = criterion_center_loss(features, pids)
        osm_caa_loss = criterion_osm_caa(features, pids , criterion_center_loss.centers.t() ) 
        lsr , lcns =  criterion_lsr(y1= y1, y2=y2, label = pids, T=None)
        
        loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss + var  + lsr_weight * lsr + lcn_weight * lcns 

        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer.step()
        for param in criterion_center_loss.parameters():
            param.grad.data *= (1./cetner_loss_weight)
        optimizer_center.step()
        losses.update(loss.data.item(), pids.size(0))
        # print(model.module.gamma)
    print("Var ({:.6f}) lcns ({:.6f}) l_sr ({:.6f})  TripletLoss  ({:.6f}) OSM Loss: ({:.6f}) Total Loss {:.6f} ({:.6f})  ".format(var.item(), lcns.item() , lsr.item() ,triplet_loss.item(), osm_caa_loss.item(),losses.val, losses.avg))        

        



def test_rerank(model, queryloader, galleryloader, pool, use_gpu):
    model.eval()
    global temp_count
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            # print (imgs.size())
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            b, n, s, c, h, w = imgs.size()
            assert(b==1)
            
            imgs = imgs.view(b*n, s, c, h, w)
            
            features = model(imgs)
            features = features.data.cpu()
            
            features = features.mean(0)
            qf.append(features.squeeze(0))
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            b, n, s, c, h, w = imgs.size()
            # imgs = imgs.view(1,-1, c, h, w)
            # imgs = imgs.view(b*n, s , c, h, w)
            assert(b==1)
            imgs = imgs.view(b*n, s, c, h, w)
            
            features = model(imgs)  
            features = features.data.cpu()
            
            features = features.mean(0)
            gf.append(features.squeeze(0))
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.stack(gf)
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
        if 'mars' not in args.dataset and 'duke_video' not in args.dataset :
            cmc= display_results(distmat, q_pids, g_pids, q_camids, g_camids,distmat_rerank=None , rerank=False)
            print("Dataset not MARS : instead", args.dataset)
            return cmc[0]
        else:
            distmat_rerank = re_ranking(qf,gf)
            mAP , re_rank_mAP = display_results(distmat, q_pids, g_pids, q_camids, g_camids, distmat_rerank=distmat_rerank, rerank=True)
            if args.focus == "map":
                print("returning map")
                return mAP
            else:
                print("returning re-rank")
                return re_rank_mAP




def display_results(distmat, q_pids, g_pids, q_camids, g_camids,distmat_rerank=None, rerank=False , ranks=[1, 5, 10, 20]):
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    if rerank:
        print("Rerank Computing CMC and mAP")
        re_rank_cmc, re_rank_mAP = evaluate(distmat_rerank, q_pids, g_pids, q_camids, g_camids)
        print("Results ---------- ")
        print("mAP: {:.1%} vs {:.1%}".format(mAP, re_rank_mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%} vs {:.1%}".format(r, cmc[r-1], re_rank_cmc[r-1]))
        print("------------------")
        return mAP , re_rank_mAP
    else:
        print("Results ---------- ")
        if 'mars' in args.dataset :
                print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1] ))
        print("------------------")
        return cmc
    

    

version =args.name
# ResNet50ta_bt9__supervised_erase__mars_4__53_checkpoint_ep201.pth.tar

if args.mode_name != '':
    if args.name == '1':
        name =storage_dir +  "resnet/bt15_dukevideo_220_150_32_5_4_95_2_59.pth.tar"
    elif args.name == '2':
        name =storage_dir + "resnet/bt15_dukevideo_220_150_32_5_4_95_3_59.pth.tar"
    elif args.name == '3':
        name = storage_dir + "resnet/bt15_dukevideo_250_150_32_4_4_95_0_59.pth.tar"
    
    # name = args.mode_name
    # name = "ResNet50ta_bt2_supervised_erase_59_checkpoint_ep81.pth.tar"
    print("loading .... " , name)
    if use_gpu:
        checkpoint = torch.load(osp.join(name)  )
    else:
        checkpoint = torch.load(osp.join(name) , map_location=torch.device('cpu') )
    state_dict = {}
    if use_gpu:
        for key in checkpoint['state_dict']:
            if "classifier" not in  key:
                state_dict["module." + key] = checkpoint['state_dict'][key]
    else:
        for key in checkpoint['state_dict']:
            if "classifier" not in  key:
                state_dict[key] = checkpoint['state_dict'][key]
    model.load_state_dict(state_dict,  strict=False)
            
    

args.save_dir = storage_dir + "resnet/trained/"
args.name += "_" + args.dataset + "_" + str(args.heads) + "_"
is_best = 0
prev_best = 0 


print ("======================" , opt , "=========================")
print (args.arch)
print("evaluation at every 10 epochs, Highly GPU/CPU expensive process, avoid running anything in Parallel")
factor = 10
args.epochs_eval = [factor * i for i in range(int(args.max_epoch / factor)) if i * factor >= args.thresold ]
if args.thresold not in  args.epochs_eval:
    args.epochs_eval.append(args.thresold)

args.epochs_eval.append(args.max_epoch-1)
print(args.epochs_eval)
prev_best = 0 

if args.evaluate: 
    print("MODEL SAVING IS ON !!!")
    for epoch in range(0, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa)
        scheduler.step()
        if epoch in args.epochs_eval :
                rank1 = test_rerank(model, queryloader, galleryloader, args.pool, use_gpu)
                if rank1 > prev_best:
                    print("\n\n Saving the model \n\n")
                    prev_best  = rank1
                    if use_gpu:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    save_checkpoint({
                            # 'centers' : criterion_center_loss.state_dict() , 
                            'state_dict': state_dict,
                        }, is_best, osp.join(args.save_dir, args.arch+ "_" + args.name + "_"  +args.opt+ '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))            
else:
    for epoch in range(0, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa)
        scheduler.step()
        if epoch in args.epochs_eval :
                rank1 = test_rerank(model, queryloader, galleryloader, args.pool, use_gpu)
                if rank1 > prev_best:
                    prev_best  = rank1
                    print("\nBest Model so far\n")


# for epoch in range(0, args.max_epoch):
#     print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
#     train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa)
#     scheduler.step()
#     if epoch in args.epochs_eval :
#             # rank1 = test_rerank(model, queryloader, galleryloader, args.pool, use_gpu)
#             # if rank1 > prev_best:
#             print("\n\n Saving the model \n\n")
#             if use_gpu:
#                 state_dict = model.module.state_dict()
#             else:
#                 state_dict = model.state_dict()

#             fpath = "/scratch/pp1953/resnet_"+ str(version) +"_" +str(args.opt)+"/" + str(args.arch)+ "_" + str(args.height) + "_" + str(args.width) + "_"  + str(args.train_batch) + "_" + str(args.num_instances) + "_" + str(args.seq_len) + "_"  + str(epoch+1) +  '.pth.tar'
#             torch.save({'state_dict': state_dict}, fpath)    

# python conf_file_super_erase.py -d=ilidsvid --opt=32 --thresold=0 --heads=10 --max-epoch=500 -a="ResNet50ta_bt11" --sampling="intelligent"                