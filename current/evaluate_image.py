from __future__ import print_function, absolute_import
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


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
from tools.eval_metrics import evaluate , re_ranking
from tools.video_loader import ImageDataset 
import tools.data_manager as data_manager
from tools.image_eval import eval, load_distribution , fliplr, eval_vehicleid

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
parser.add_argument('--test-batch', default=256, type=int)
# Architecture
parser.add_argument('-a', '--arch', type=str, default="ResNet50ta_bt5", help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=int, default=1)
parser.add_argument('-n', '--mode-name', type=str, default='', help="ResNet50ta_bt2_supervised_erase_59_checkpoint_ep81.pth.tar, \
    ResNet50ta_bt2_supervised_erase_44_checkpoint_ep101.pth.tar")

# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--name', '--model_name', type=str, default='_supervised_erase_')
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-f', '--focus', type=str, default='rank-1', help="map,rerank_map")
parser.add_argument('--fin-dim', default=2048, type=int, help="final dim for center loss")
parser.add_argument('--rerank', action='store_true', help="evaluation only")







args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_gpu = torch.cuda.is_available()
np.random.seed(args.seed)

args.gpu_devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

cudnn.benchmark = True
if args.dataset == "cuhk01":
    if args.pool == 100:
        dataset = data_manager.init_dataset(name=args.dataset, splits= 100)

dataset = data_manager.init_dataset(name=args.dataset)

pin_memory = True if use_gpu else False
transform_test = transforms.Compose([
    transforms.Resize((args.height, args.width), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if args.dataset == "market2":
    args.focus ="map"


if args.dataset == "market":
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
lamb = 0.3

print("==========")
attention_heads = None 
model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids , fin_dim=args.fin_dim, heads=args.heads)
# model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids )
# if mode == 5:
#     path = storage_dir +"resnet/" +"bt10_vehicle_78_150_250_44.pth.tar"
#     print( "Loading vehicleid model" )
# elif mode == 7:
#     path = storage_dir +"resnet/" +"bt10_veri_82_44.pth.tar"
#     print( "Loading VERI model" )
# else:
path =  storage_dir +"resnet/" +"mars_sota.pth.tar"
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

if mode != 3 and mode != 7 :
    path = "/beegfs/pp1953/distribution_" + args.dataset +  ".mat"
    distribution = load_distribution(path= path , dataset=args.dataset)
    print("distribution %s loaded"%(path))
    


if use_gpu:
    model = nn.DataParallel(model).cuda()


def test_rerank(model, queryloader, galleryloader , use_gpu=True):
    model.eval()
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
    cam_dis_query = None
    cam_dis_gallery = None
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
                        img = fliplr(img)
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
            if mode != 3 and mode != 7 :
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
                        img = fliplr(img)
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
        elif mode == 3:
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.numpy()
            gf = gf.numpy()
            qf = qf.numpy()
            # distmat_rerank = re_ranking(qf,gf)
            distmat_rerank = None
            mAP , re_rank_mAP , CMC = display_results(distmat, q_pids, g_pids, q_camids, g_camids,distmat_rerank, rerank=False )
        else:
            gf=  gf.numpy()
            qf=  qf.numpy()
            
            gf=gf.transpose()/np.power(np.sum(np.power(gf,2),axis=1),0.5)
            gf=gf.transpose()
            qf=qf.transpose()/np.power(np.sum(np.power(qf,2),axis=1),0.5)
            qf=qf.transpose()
            if type(cam_dis_query) == type(None):
                CMC, mAP =  eval(g_pids , q_pids , qf,  q_camids, qframes, gf , g_camids ,gframes , distribution  )
            else:
                CMC, mAP =  eval(g_pids , q_pids , qf,  q_camids, qframes, gf , g_camids ,gframes , distribution , cam_dis_query=cam_dis_query, cam_dis_gallery=cam_dis_gallery )
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
    


args.save_dir = storage_dir 
# args.save_dir = storage_dir + "resnet/"
args.name += "_" + args.dataset + "_" + str(args.heads) + "_"
is_best = 0
prev_best = 0 


# if args.101mode_name != '':
name = "resnet/bt13_cuhk01_100_220_150_32_4_100_44.pth.tar"

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

del state_dict, checkpoint


rank1 = test_rerank(model, queryloader, galleryloader, use_gpu)





# python evaluate_image.py -d=$dataset -a="ResNet50ta_bt13" --sampling="intelligent"  --height=$height --width=$width --seq-len=$s