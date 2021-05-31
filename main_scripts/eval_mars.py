import scipy.io
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

# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--name', '--model_name', type=str, default='_supervised_erase_')
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-s', '--sampling', type=str, default='random', help="choose sampling for training")

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



def test_rerank2(model, queryloader, galleryloader, use_gpu):
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
        gf=  gf.numpy()
        qf=  qf.numpy()
        
        result = {'gallery_f':gf,'gallery_label':g_pids,'gallery_cam':g_camids,'query_f':qf,'query_label':q_pids,'query_cam':q_camids}
        scipy.io.savemat(storage_dir + args.dataset +'_result.msat',result)
        
        # m, n = qf.size(0), gf.size(0)
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.numpy()
        # gf1 = gf.numpy()
        # qf1 = qf.numpy()
        # display_results(distmat, q_pids, g_pids, q_camids, g_camids,distmat_rerank, rerank=True )
        # gf2 = gf.numpy()
        # qf2 = qf.numpy()
        gf=gf.transpose()/np.power(np.sum(np.power(gf,2),axis=1),0.5)
        gf=gf.transpose()
        qf=qf.transpose()/np.power(np.sum(np.power(qf,2),axis=1),0.5)
        qf=qf.transpose()
        score = -np.matmul(qf , gf.transpose())
        # distmat_rerank = re_ranking(qf2,gf2)
        display_results(score, q_pids, g_pids, q_camids, g_camids,distmat_rerank=None, rerank=False )




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
    

path = storage_dir + "resnet/bt15_mars_250_150_32_5_4_86_1_53.pth.tar"
print("loading .... " , path)
state_dict = {}
if use_gpu:
    checkpoint = torch.load(path)
    for key in checkpoint['state_dict']:
        state_dict["module." + key] = checkpoint['state_dict'][key]
else:
    checkpoint = torch.load(path , map_location=torch.device('cpu') )
    for key in checkpoint['state_dict']:
        state_dict[key] = checkpoint['state_dict'][key]

model.load_state_dict(state_dict,  strict=True)
print("LOADED :D :D :D ")
test_rerank2(model, queryloader, galleryloader, use_gpu)



    
# python eval_mars.py -d=mars -a="ResNet50ta_bt15" --height=250 --width=150 