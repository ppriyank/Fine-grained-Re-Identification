
import os
import sys
import glob
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
import models

from tools.eval_metrics import evaluate , re_ranking
from tools.video_loader import VideoDataset 
import tools.data_manager as data_manager
from tools.image_eval import fliplr
from tools.image_eval import eval, load_distribution , fliplr, scoring , eval2 , eval_vehicleid

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
parser.add_argument('--heads', default=1, type=int, help="no of heads of multi head attention")
parser.add_argument('--fin-dim', default=2048, type=int, help="final dim for center loss")


args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_gpu = torch.cuda.is_available()
np.random.seed(args.seed)

args.gpu_devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
cudnn.benchmark = True
dataset = data_manager.init_dataset(name=args.dataset)

pin_memory = True if use_gpu else False

transform_test = transforms.Compose([
    transforms.Resize((args.height, args.width), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


queryloader = DataLoader(
    VideoDataset(dataset.query, seq_len=args.seq_len, sample='test', transform=transform_test, max_length=40),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)


galleryloader = DataLoader(
    VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='test', transform=transform_test , max_length=40),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)

print(args)
lamb = 0.3
model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids , fin_dim=args.fin_dim)
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
if use_gpu:
    model = nn.DataParallel(model).cuda()


def test_rerank1(model, queryloader, galleryloader, pool, use_gpu):
    print("Evaluating....")
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
            features.squeeze(0)            
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
        if 'mars' not in args.dataset :
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


def test_rerank2(model, queryloader, galleryloader, pool, use_gpu):
    model.eval()
    global temp_count
    q_pids, q_camids = [], []
    qf = torch.FloatTensor()
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            # print (imgs.size())
            imgs = Variable(imgs)
            b, n, s, c, h, w = imgs.size()
            assert(b==1)
            imgs = imgs.view(b*n, s, c, h, w)
            ff = torch.FloatTensor(n,2048).zero_() 
            for i in range(2):
                if(i==1):
                    for j in range(s):
                        imgs[:,j,:] = fliplr(imgs[:,j,:])
                if use_gpu:
                    imgs = imgs.cuda()
                outputs = model(imgs) 
                f = outputs.data.cpu()
                ff = ff+f
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            ff = ff.mean(0).unsqueeze(0)
            qf = torch.cat((qf,ff), 0)
            
            q_pids.extend(pids)
            q_camids.extend(camids)
            
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        
        g_pids, g_camids =  [], []
        gf = torch.FloatTensor()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            b, n, s, c, h, w = imgs.size()
            assert(b==1)
            imgs = imgs.view(b*n, s, c, h, w)
            ff = torch.FloatTensor(n,2048).zero_() 
            for i in range(2):
                if(i==1):
                    for j in range(s):
                        imgs[:,j,:] = fliplr(imgs[:,j,:])
                if use_gpu:
                    imgs = imgs.cuda()
                outputs = model(imgs) 
                f = outputs.data.cpu()
                ff = ff+f
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            ff = ff.mean(0).unsqueeze(0)
            gf = torch.cat((gf,ff), 0)
            
            g_pids.extend(pids)
            g_camids.extend(camids)

        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()
        gf = gf.numpy()
        qf = qf.numpy()

        print("computing cos ranks")
        CMC_dot, mAP_dot = eval_vehicleid(q_pids, g_pids, q_camids, g_camids, qf, gf,max_rank=21)
        print(mAP_dot, CMC_dot[0], CMC_dot[4], CMC_dot[9], CMC_dot[19])
        print("computing L2 ranks")
        CMC, mAP = eval_vehicleid(q_pids, g_pids, q_camids, g_camids, qf, gf,max_rank=21, pre_compute_score=distmat, reverse=False)
        print(mAP, CMC[0], CMC[4], CMC[9], CMC[19])
        print("computing re ranks")
        distmat_rerank = re_ranking(qf,gf)
        CMC_re_rank, mAP_re_rank = eval_vehicleid(q_pids, g_pids, q_camids, g_camids, qf, gf,max_rank=21, pre_compute_score=distmat_rerank, reverse=False)
        
        ranks=[1, 5, 10, 20]
        print("Results ---------- ")
        print("mAP: {:.1%} vs {:.1%} vs {:.1%}".format(mAP_dot, mAP, mAP_re_rank))
        print("CMC curve :: dot vs l2 vs re re-rank")
        for r in ranks:
            print("Rank-{:<3}: {:.1%} vs {:.1%} vs {:.1%}".format(r, CMC_dot[r-1], CMC[r-1], CMC_re_rank[r-1]))
        print("------------------")


        if 'mars' not in args.dataset :
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
    


path= "/scratch/pp1953/resnet_" + args.opt + "/"
for file in glob.glob(path + "*.pth.tar"):
    print(file)
    print("loading .... " , file)
    state_dict = {}
    if use_gpu:
        checkpoint = torch.load(file)
        for key in checkpoint['state_dict']:
            state_dict["module." + key] = checkpoint['state_dict'][key]
    else:
        checkpoint = torch.load(file , map_location=torch.device('cpu') )
        for key in checkpoint['state_dict']:
            state_dict[key] = checkpoint['state_dict'][key]
    
    model.load_state_dict(state_dict,  strict=True)
    print("LOADED :D :D :D ")
    test_rerank1(model, queryloader, galleryloader, args.pool, use_gpu)



    
# python evaluate.py -d=mars -a="ResNet50ta_bt15" --opt=39 --height=200 --width=150 