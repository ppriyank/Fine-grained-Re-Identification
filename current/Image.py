from __future__ import print_function, absolute_import
import os
import sys

# Do not run these if you in terminal
currentdir = os.getcwd() 
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

storage_dir = "/scratch/pp1953/"
pretrained_ResNets = storage_dir +"resnet/"


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
import models

from loss import CrossEntropyLabelSmooth, TripletLoss , CenterLoss , OSM_CAA_Loss , Satisfied_Rank_loss2
from tools.transforms2 import *
from tools.scheduler import WarmupMultiStepLR
from tools.utils import AverageMeter, save_checkpoint
from tools.eval_metrics import evaluate , re_ranking
from tools.samplers import RandomIdentitySampler
from tools.video_loader import ImageDataset , Image_inderase
import tools.data_manager as data_manager
from tools.image_eval import eval, load_distribution , fliplr, eval_vehicleid


print("Current File Name : ",os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Train Image model')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet (Number of images belonging to the same label)")
# Optimization options
parser.add_argument('--max-epoch', default=500, type=int,
                    help="maximum epochs to run")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=256, type=int)
# Architecture
parser.add_argument('-a', '--arch', type=str, default="ResNet50TA_BT_image", help="ResNet50TA_BT_image or ResNet50TA_BT_video")
parser.add_argument('--split', type=int, default=100)


# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-opt', '--opt', type=str, default='dataset', help="choose configuration setting")
parser.add_argument('--thresold', type=int, default='60')

parser.add_argument('-n', '--mode-name', type=str, default='', help="ResNet50ta_bt2_supervised_erase_59_checkpoint_ep81.pth.tar, \
    ResNet50ta_bt2_supervised_erase_44_checkpoint_ep101.pth.tar")

parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--save-dir', type=str, default=pretrained_ResNets + "trained/")
parser.add_argument('--pretrain', action='store_true', help="evaluation only")
parser.add_argument('--fin-dim', default=2048, type=int, help="final dim for center loss")


parser.add_argument('-f', '--focus', type=str, default='rank-1', help="map,rerank_map")
parser.add_argument('--rerank', action='store_true', help="evaluation only")
parser.add_argument('--load_distribution', action='store_true', help="evaluation only")


args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_gpu = torch.cuda.is_available()
np.random.seed(args.seed)


try:
    os.makedirs(args.save_dir)
except FileExistsError:
    # directory already exists
    pass
    
if use_gpu:
    print("train_batch===" ,  args.train_batch , "seq_len" , args.seq_len, "no of gpus : " , os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count() )
else:
    print("train_batch===" ,  args.train_batch , "seq_len" , args.seq_len)

args.gpu_devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

cudnn.benchmark = True
if args.dataset == "cuhk01":
    print("  SPLIT --  ", args.split)
    dataset = data_manager.init_dataset(name=args.dataset, splits= args.split)
else:
    dataset = data_manager.init_dataset(name=args.dataset)

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
    Image_inderase(dataset.train, seq_len=args.seq_len, transform=transform_train , height=args.height, width=args.width),
    batch_size=args.train_batch, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=True,
)

if args.dataset == "market2":
    args.focus ="map"
    mode = 1
elif args.dataset == "market":
    mode = 1
    args.focus ="map"
elif args.dataset == "duke":
    mode =  2
elif args.dataset == "cuhk03":
    mode =  4
elif args.dataset == "veri":
    mode =  5    
    args.focus ="map"
elif args.dataset == "vric":
    mode =  6   
elif args.dataset == "vehicleid":
    mode =  7  
else:
    mode =  3


queryloader = DataLoader(
    ImageDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test,eval=True, mode=mode, height=args.height, width=args.width),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)

galleryloader = DataLoader(
    ImageDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test,eval=True, mode=mode , height=args.height, width=args.width),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)

print(args)
opt = args.opt
lamb = 0.3
base_learning_rate = 0.00035
print("==========")
config = configparser.ConfigParser()
print("dataset_config.conf")
config.read(currentdir + "/../tools/dataset_config.conf")        
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

print("==========")
model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids , fin_dim=args.fin_dim)

if args.pretrain:
    print("LOADING PRETRAINED ResNet")
    if mode == 5:
        path =  pretrained_ResNets +"bt13_vehicle_150_250_32_4_78_8_39.pth.tar"
        print( "Loading vehicleid model" )
    elif mode == 7:
        path = pretrained_ResNets +"bt10_veri_82_44.pth.tar"
        print( "Loading VERI model" )
    else:
        path =  pretrained_ResNets +"mars_sota.pth.tar"
        print( "Loading MARS model" , path)
    if use_gpu:
        checkpoint = torch.load( path  )
    else:
        checkpoint = torch.load( path , map_location=torch.device('cpu') )
    state_dict = {}
    state_dict2 = {}
    if mode == 5 :
        for key in checkpoint['state_dict']:
            if "base_mars" in  key :
                temp = key.replace("base_mars." , "")
                state_dict[temp] = checkpoint['state_dict'][key]
    else:
        for key in checkpoint['state_dict']:
            if "base" in  key :
                temp = key.replace("base." , "")
                state_dict[temp] = checkpoint['state_dict'][key]
    model.base_mars.load_state_dict(state_dict,  strict=True)
    del  state_dict , state_dict2, checkpoint


print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

if args.load_distribution:
    print("LOADING DISTRIBUTION")
    if mode != 3 and mode != 7 :
        path = "/beegfs/pp1953/distribution_" + args.dataset +  ".mat"
        distribution = load_distribution(path= path , dataset=args.dataset)
        print("distribution %s loaded"%(path))

    
criterion_htri = TripletLoss(margin, 'cosine')
criterion_xent = CrossEntropyLabelSmooth(dataset.num_train_pids,use_gpu=use_gpu)
criterion_center_loss = CenterLoss(use_gpu=use_gpu , feat_dim=args.fin_dim , num_classes= dataset.num_train_pids)
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
    for batch_idx, (imgs, pids) in enumerate(trainloader):
        pids = pids.view(-1)
        imgs  = imgs.view(-1, 3, imgs.shape[3], imgs.shape[4])
        if use_gpu:
            imgs, pids   = imgs.cuda(), pids.cuda() 
        imgs, pids = Variable(imgs), Variable(pids)
        y1,y2 , features  =  model(imgs)
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
        
        # loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight  + var  + lsr_weight * lsr + lcn_weight * lcns 
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
    print("Var ({:.6f}) lcns ({:.6f}) l_sr ({:.6f})  TripletLoss  ({:.6f}) OSM Loss: ({:.6f}) Total Loss {:.6f} ({:.6f})  ".format(var.item(),  lcns.item() , lsr.item() ,triplet_loss.item(), osm_caa_loss.item(),losses.val, losses.avg))        

        



def test_rerank(model, queryloader, galleryloader , use_gpu=True):
    model.eval()
    normal = True
    if mode == 3 or mode == 7 :
        normal = True
    else:
        normal = False
    qframes = np.array([])
    gframes = np.array([])
    q_camids = []
    q_pids = []

    g_camids = []
    g_pids = []    
    qf = torch.FloatTensor()
    with torch.no_grad():
        for batch_idx, (imgs) in enumerate(queryloader):
            if mode != 3 and mode != 7 :
                img, pid, camid , fname= imgs    
                qframes = np.append(qframes, fname.numpy())
            else:
                img, pid, camid  = imgs
            q_camids.extend(camid)
            q_pids.extend(pid)
            n, c, h, w = img.size()
            if normal:
                input_img = Variable(img)
                if use_gpu:
                        input_img = input_img.cuda()
                outputs = model(input_img) 
                ff = outputs.data.cpu()
            else:
                ff = torch.FloatTensor(n,2048).zero_() 
                for i in range(2):
                    if(i==1):
                        img = fliplr(img, False)
                    input_img = Variable(img)
                    if use_gpu:
                        input_img = input_img.cuda()
                    outputs = model(input_img) 
                    f = outputs.data.cpu()
                    ff = ff+f
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            qf = torch.cat((qf,ff), 0)

        q_camids = np.asarray(q_camids)
        q_pids = np.asarray(q_pids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        # gf, g_pids, g_camids , g_frames = [], [], [] , []
        gf = torch.FloatTensor()
        for batch_idx, (imgs) in enumerate(galleryloader):
            if mode != 3 and mode != 7  :
                img, pid, camid , fname= imgs  
                gframes = np.append(gframes, fname.numpy())
            else:
                img, pid, camid  = imgs
            
            g_camids.extend(camid)
            g_pids.extend(pid)
            n, c, h, w = img.size()
            if normal:
                input_img = Variable(img)
                if use_gpu:
                        input_img = input_img.cuda()
                outputs = model(input_img) 
                ff = outputs.data.cpu()
            else:
                ff = torch.FloatTensor(n,2048).zero_() 
                for i in range(2):
                    if(i==1):
                        img = fliplr(img,  False)
                    input_img = Variable(img)
                    if use_gpu:
                        input_img = input_img.cuda()
                    outputs = model(input_img) 
                    f = outputs.data.cpu()
                    ff = ff+f
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            gf = torch.cat((gf,ff), 0)

        g_camids = np.asarray(g_camids)
        g_pids = np.asarray(g_pids)
        
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")
        m, n = qf.size(0), gf.size(0)
        # result = {'gallery_f':gf,'gallery_label':g_pids,'gallery_cam':g_camids,'gallery_frames':gframes,'query_f':qf,'query_label':q_pids,'query_cam':q_camids,'query_frames':qframes}
        # scipy.io.savemat('pytorch_result4.mat',result)
        if mode == 7: 
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.numpy()
            gf = gf.numpy()
            qf = qf.numpy()
            # distmat_rerank = re_ranking(qf,gf)
            distmat_rerank = None
            CMC, mAP = eval_vehicleid(q_pids, g_pids, q_camids, g_camids, qf, gf,max_rank=21, pre_compute_score=distmat, reverse=False)
            # CMC, mAP = eval_vehicleid(q_pids, g_pids, q_camids, g_camids, qf, gf,max_rank=21)
            print("Results ---------- ")
            print("mAP: {:.1%}".format(mAP))
            print("CMC curve")
            ranks=[1, 5, 10, 20]
            for r in ranks:
                print("Rank-{:<3}: {:.3%}".format(r, CMC[r-1] ))
            print("------------------")
        elif mode == 3 or mode == 1 or mode == 5:
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.numpy()
            gf = gf.numpy()
            qf = qf.numpy()
            distmat_rerank = re_ranking(qf,gf)
            # distmat_rerank = None
            mAP , re_rank_mAP , CMC = display_results(distmat, q_pids, g_pids, q_camids, g_camids,distmat_rerank, rerank=True )
        else:
            gf=  gf.numpy()
            qf=  qf.numpy()
            
            gf=gf.transpose()/np.power(np.sum(np.power(gf,2),axis=1),0.5)
            gf=gf.transpose()
            qf=qf.transpose()/np.power(np.sum(np.power(qf,2),axis=1),0.5)
            qf=qf.transpose()
            CMC, mAP =  eval(g_pids , q_pids , qf,  q_camids, qframes, gf , g_camids ,gframes , distribution  )
            ranks=[1, 5, 10, 20]
            print("Results ---------- ")
            print("mAP: {:.1%}".format(mAP))
            print("CMC curve")
            for r in ranks:
                print("Rank-{:<3}: {:.1%}".format(r, CMC[r-1]))
            print("------------------")
        
        if args.focus == "map":
            print("returning map")
            return mAP
        else:
            print("returning rank-1")
            return CMC[0]
        
        



def display_results(distmat, q_pids, g_pids, q_camids, g_camids,distmat_rerank=None, rerank=False , ranks=[1, 5, 10, 20]):
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    if rerank :
        print("Rerank Computing CMC and mAP")
        re_rank_cmc, re_rank_mAP = evaluate(distmat_rerank, q_pids, g_pids, q_camids, g_camids)
        print("Results ---------- ")
        print("mAP: {:.1%} vs {:.1%}".format(mAP, re_rank_mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%} vs {:.1%}".format(r, cmc[r-1], re_rank_cmc[r-1]))
        print("------------------")
        return mAP , re_rank_mAP , cmc
    else:
        print("Results ---------- ")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1] ))
        print("------------------")
        return mAP , None , cmc
    

 
if args.mode_name != '':
    name=args.mode_name
    print("loading pre-trained model .... " , name)
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
        # for key in checkpoint['state_dict']:
        #     state_dict["module." + key] = checkpoint['state_dict'][key]
        # model.load_state_dict(state_dict,  strict=True)
    del state_dict, checkpoint



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
    print("Evalauting the model as well :D ") 
    for epoch in range(0, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa)
        scheduler.step()
        if epoch in args.epochs_eval :
                rank1 = test_rerank(model, queryloader, galleryloader, use_gpu)
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
                        }, osp.join(args.save_dir,  args.arch+ "_" + args.dataset + "_" + args.opt+"_" + str(args.height) + "_" + str(args.width)+ "_" + str(args.seq_len) + "_" + str(args.train_batch) + '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))            
else:
    for epoch in range(0, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa)
        scheduler.step()
        if epoch in args.epochs_eval :
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                save_checkpoint({
                            # 'centers' : criterion_center_loss.state_dict() , 
                            'state_dict': state_dict,
                }, osp.join(args.save_dir,  args.arch+ "_" + args.dataset + "_" + args.opt+"_" + str(args.height) + "_" + str(args.width)+ "_" + str(args.seq_len) + "_" + str(args.train_batch) + '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))            

