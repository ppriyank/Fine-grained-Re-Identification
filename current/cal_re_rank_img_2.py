from __future__ import print_function, absolute_import
import os
import sys
import scipy.io
# currentdir = os.path.dirname(os.path.realpath("/home/pp1953/code/Video-Person-ReID-master/current/conf_file_super_erase_image.py"))
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

storage_dir = "/scratch/pp1953/"
import argparse
import random 
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
import models
from tools.eval_metrics import evaluate , re_ranking
from tools.video_loader import ImageDataset 
import tools.data_manager as data_manager
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
parser.add_argument('--test-batch', default=256, type=int)
parser.add_argument('--num-instances', type=int, default=4, help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default="ResNet50ta_bt5", help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=int, default=1)
parser.add_argument('-n', '--mode-name', type=str, default='', help="ResNet50ta_bt2_supervised_erase_59_checkpoint_ep81.pth.tar, \
    ResNet50ta_bt2_supervised_erase_44_checkpoint_ep101.pth.tar")

# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-s', '--sampling', type=str, default='random', help="choose sampling for training")
parser.add_argument('-f', '--focus', type=str, default='rank-1', help="map,rerank_map")
parser.add_argument('--fin-dim', default=2048, type=int, help="final dim for center loss")
parser.add_argument('--rerank', action='store_true', help="evaluation only")
parser.add_argument('--normal', action='store_true', help="evaluation only")


args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_gpu = torch.cuda.is_available()
args.gpu_devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


if args.dataset == "cuhk01":
    if args.pool == 100:
        dataset = data_manager.init_dataset(name=args.dataset, splits= 100)

dataset = data_manager.init_dataset(name=args.dataset, test_size=1600)

pin_memory = True if use_gpu else False

transform_test = transforms.Compose([
    transforms.Resize((args.height, args.width), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



if args.dataset == "market" or args.dataset == "market2":
    mode = 1
elif args.dataset == "duke":
    mode =  2
elif args.dataset == "cuhk03":
    mode =  4
elif args.dataset == "veri":
    mode =  5 
    # mode =  3    
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

model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids , fin_dim=args.fin_dim)
# model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids )
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

if mode != 3 and mode != 7 :
    path = "/beegfs/pp1953/distribution_" + args.dataset +  ".mat"
    distribution = load_distribution(path= path , dataset=args.dataset)
    print("distribution %s loaded"%(path))

if use_gpu:
    model = nn.DataParallel(model).cuda()



def test_rerank(model, queryloader, galleryloader , use_gpu=True):
    model.eval()
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
            if mode != 3  and mode !=7 :
                img, pid, camid , fname= imgs    
                qframes = np.append(qframes, fname.numpy())
            else:
                img, pid, camid  = imgs
            q_camids.extend(camid)
            q_pids.extend(pid)
            n, c, h, w = img.size()
            img = Variable(img)
            if use_gpu:
                    img = img.cuda()
            if args.normal:
                outputs = model(img) 
                ff = outputs.data.cpu()
            else:
                ff = torch.FloatTensor(n,2048).zero_() 
                for i in range(2):
                    if(i==1):
                        img = fliplr(img)
                    outputs = model(img) 
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
            if mode != 3 and mode !=7 :
                img, pid, camid , fname= imgs  
                gframes = np.append(gframes, fname.numpy())
            else:
                img, pid, camid  = imgs
            g_camids.extend(camid)
            g_pids.extend(pid)
            n, c, h, w = img.size()
            img = Variable(img)
            if use_gpu:
                    img = img.cuda()
            if args.normal:
                outputs = model(img) 
                ff = outputs.data.cpu()
            else:
                ff = torch.FloatTensor(n,2048).zero_() 
                for i in range(2):
                    if(i==1):
                        img = fliplr(img)
                    outputs = model(img) 
                    f = outputs.data.cpu()
                    ff = ff+f
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            gf = torch.cat((gf,ff), 0)
        g_camids = np.asarray(g_camids)
        g_pids = np.asarray(g_pids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("saving weights")
        gf=  gf.numpy()
        qf=  qf.numpy()
        
        result = {'gallery_f':gf,'gallery_label':g_pids,'gallery_cam':g_camids,'gallery_frames':gframes,'query_f':qf,'query_label':q_pids,'query_cam':q_camids,'query_frames':qframes}
        scipy.io.savemat(storage_dir + args.dataset +'_result.msat',result)
        del result
        re_score(gf, g_pids , g_camids, gframes, qf , q_pids, q_camids, qframes)

        

def re_score(gf, g_pids , g_camids, gframes, qf , q_pids, q_camids, qframes):
    print("Computing distance matrix")
    print("------------------""------------------""------  MODE ::: 7 ------------------""------------------""------------------")
    qf = torch.tensor(qf)
    gf = torch.tensor(gf)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()

    # print("computing cos ranks")
    # CMC_dot, mAP_dot = eval_vehicleid(q_pids, g_pids, q_camids, g_camids, qf, gf,max_rank=21)
    # print(mAP_dot, CMC_dot[0], CMC_dot[4], CMC_dot[9], CMC_dot[19])
    # print("computing L2 ranks")
    # CMC, mAP = eval_vehicleid(q_pids, g_pids, q_camids, g_camids, qf, gf,max_rank=21, pre_compute_score=distmat, reverse=False)
    # print(mAP, CMC[0], CMC[4], CMC[9], CMC[19])
    # print("computing re ranks")
    distmat_rerank = re_ranking(qf,gf)
        
    # CMC_re_rank, mAP_re_rank = eval_vehicleid(q_pids, g_pids, q_camids, g_camids, qf, gf,max_rank=21, pre_compute_score=distmat_rerank, reverse=False)
    # print(mAP_re_rank, CMC_re_rank[0], CMC_re_rank[4], CMC_re_rank[9], CMC_re_rank[19])
    # ranks=[1, 5, 10, 20]
    # print("Results ---------- ")
    # print("mAP: {:.1%} vs {:.1%} vs {:.1%}".format(mAP_dot, mAP, mAP_re_rank))
    # print("CMC curve :: dot vs l2 vs re re-rank")
    # for r in ranks:
    #     print("Rank-{:<3}: {:.1%} vs {:.1%} vs {:.1%}".format(r, CMC_dot[r-1], CMC[r-1], CMC_re_rank[r-1]))
    print("------------------""------------------""------------------""------------------""------------------""------------------")
    print("------------------""------------------""------  MODE ::: 3 ------------------""------------------""------------------")    
    mAP , re_rank_mAP , CMC = display_results(distmat, q_pids, g_pids, q_camids, g_camids,distmat_rerank, rerank=True )
    gf=gf.transpose()/np.power(np.sum(np.power(gf,2),axis=1),0.5)
    gf=gf.transpose()
    qf=qf.transpose()/np.power(np.sum(np.power(qf,2),axis=1),0.5)
    qf=qf.transpose()
    distmat_rerank = re_ranking(qf,gf)
    score = -np.matmul(qf , gf.transpose())
    mAP , re_rank_mAP , CMC = display_results(score, q_pids, g_pids, q_camids, g_camids,distmat_rerank, rerank=True )
    print("------------------""------------------""------------------""------------------""------------------""------------------")
    print("------------------""------------------""------  MODE ::: 1 ------------------""------------------""------------------")    
    CMC_orig, mAP_orig =  eval(g_pids , q_pids , qf,  q_camids, qframes, gf , g_camids ,gframes , distribution  )
    ranks=[1, 5, 10, 20]
    print("Results ---------- ")
    print("mAP: {:.1%}".format(mAP_orig))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, CMC_orig[r-1]))
    print("------------------")
    all_features = np.concatenate([qf,gf],axis=0)
    all_labels = np.concatenate([q_pids,g_pids],axis=0)
    all_cams = np.concatenate([q_camids,g_camids],axis=0)
    all_frames = np.concatenate([qframes,gframes],axis=0)
    all_scores = np.zeros((len(all_labels),len(all_labels)))
    print('all_features shape:',all_features.shape)
    print('all_labels shape:',all_labels.shape)
    print('all_cams shape:',all_cams.shape)
    print('all_frames shape:',all_frames.shape)
    print('all_scores shape:',all_scores.shape)     
    CMC = torch.IntTensor(len(all_labels)).zero_()
    ap = 0.0        
    for i in range(len(all_labels)):
        scores_new = scoring(all_features[i],all_labels[i],all_cams[i],all_frames[i], all_features,all_labels,all_cams,all_frames,distribution)
        # print('scores_new shape:',scores_new.shape)
        all_scores[i,:] = scores_new

    print('type(all_scores):',type(all_scores))
    all_scores = {'all_scores':all_scores}
    scipy.io.savemat(storage_dir + args.dataset + '_all_scores.mat',all_scores)
    all_dist = all_scores["all_scores"]
    CMC, mAP = eval2(g_pids , q_pids , qf,  q_camids, qframes, gf , g_camids ,gframes , all_dist)
    ranks=[1, 5, 10, 20]
    print("Results ---------- ")
    print("mAP: {:.1%} , {:.1%}".format(mAP_orig, mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}, {:.1%}".format(r, CMC_orig[r-1], CMC[r-1]))
    print("------------------")
    

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
        if 'mars' in args.dataset :
                print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1] ))
        print("------------------")
        return mAP , None , cmc
    


args.save_dir = storage_dir 
# args.save_dir = storage_dir + "resnet/"
# name =  "resnet/bt10_veri_82_44.pth.tar"
# name =  "resnet/bt10_vehicle_75_112_224.pth.tar"
# name =  "resnet/bt10_vehicle_76_7_150_250.pth.tar"
# names =  ["resnet/bt15_market2_250_150_32_4_91_3_44.pth.tar", "bt15_market_250_150_32_4_86_1_55.pth.tar" , 
#     "bt15_market2_250_150_32_4_91_0_39.pth.tar", 
#     "bt15_market2_250_150_32_4_91_0_54.pth.tar", "bt13_market_250_150_32_4_92_5_44.pth.tar"]

# names =  ["resnet/bt13_market_250_150_32_4_92_5_44.pth.tar"]
names =  ["resnet/bt13_veri_150_250_32_4_84_2_39.pth.tar"]
# names =  ["resnet/bt13_veri_150_250_32_4_84_0_39.pth.tar"]

for name in names : 
    for i in range(2):
        try:
            if i == 0 :
                args.normal = True
                print("normal is On :D")
            else:
                args.normal = False
                print("normal is Off :( ")
            print("loading .... " , name)
            if use_gpu:
                checkpoint = torch.load(osp.join(args.save_dir, name)  )
            else:
                checkpoint = torch.load(osp.join(args.save_dir, name) , map_location=torch.device('cpu') )
            state_dict = {}
            if use_gpu:
                for key in checkpoint['state_dict']:
                        state_dict["module." + key] = checkpoint['state_dict'][key]
            else:
                for key in checkpoint['state_dict']:
                        state_dict[key] = checkpoint['state_dict'][key]
            model.load_state_dict(state_dict,  strict=True)
            del state_dict, checkpoint
            rank1 = test_rerank(model, queryloader, galleryloader, use_gpu)
        except Exception as e:
            print("FAILED " , name, i)
            continue 

# for name in names : 
#     args.normal = False
#     print("normal is Off :( ")
#     print("loading .... " , name)
#     if use_gpu:
#         checkpoint = torch.load(osp.join(args.save_dir, name)  )
#     else:
#         checkpoint = torch.load(osp.join(args.save_dir, name) , map_location=torch.device('cpu') )
#     state_dict = {}
#     if use_gpu:
#         for key in checkpoint['state_dict']:
#                 state_dict["module." + key] = checkpoint['state_dict'][key]
#     else:
#         for key in checkpoint['state_dict']:
#                 state_dict[key] = checkpoint['state_dict'][key]
#     model.load_state_dict(state_dict,  strict=True)
#     del state_dict, checkpoint
#     rank1 = test_rerank(model, queryloader, galleryloader, use_gpu)


# import pdb
# pdb.set_trace()
# file = storage_dir + args.dataset +'_result.msat'
# result = scipy.io.loadmat(file)

# gf = result['gallery_f']
# qf = result['query_f']
# g_pids = result['gallery_label']
# g_camids = result['gallery_cam']
# gframes = result['gallery_frames']

# q_pids = result['query_label']
# q_camids = result['query_cam']
# qframes = result['query_frames']
# g_pids = g_pids[0]
# g_camids = g_camids[0]
# q_pids = q_pids[0]
# q_camids = q_camids[0]
# gframes = gframes[0]
# qframes = qframes[0] 
# if not osp.exists(file):
#         rank1 = test_rerank(model, queryloader, galleryloader, use_gpu)
# else:   
    # 
#     
#     
#     if mode != 3 and mode != 7 :
#         
#         
#     file = storage_dir + args.dataset + '_all_scores.mat'
#     if not osp.exists(file):
#         re_score(gf, g_pids , g_camids, gframes, qf , q_pids, q_camids, qframes)
#     else:
#         all_scores = scipy.io.loadmat(file)                   
#         all_dist =  all_scores['all_scores']
#         print('all_dist shape:',all_dist.shape)
#         CMC_orig, mAP_orig =  eval(g_pids , q_pids , qf,  q_camids, qframes, gf , g_camids ,gframes , distribution  )
#         CMC, mAP = eval2(g_pids , q_pids , qf,  q_camids, qframes, gf , g_camids ,gframes , all_dist)

        
#         ranks=[1, 5, 10, 20]
#         print("Results ---------- ")
#         print("mAP: {:.1%} , {:.1%}".format(mAP_orig, mAP))
#         print("CMC curve")
#         for r in ranks:
#             print("Rank-{:<3}: {:.1%}, {:.1%}".format(r, CMC_orig[r-1], CMC[r-1]))
#         print("------------------")

# CMC = CMC.float()
# CMC = CMC/len(query_label) #average CMC
# print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))



       
    
# python cal_re_rank_img.py -d=veri -a="ResNet50ta_bt10" --sampling="intelligent" --height=150 --width=250
# python cal_re_rank_img.py -d=vehicleid -a="ResNet50ta_bt10" --sampling="intelligent" --height=150 --width=250
# python cal_re_rank_img.py -d=vehicleid -a="ResNet50ta_bt10" --sampling="intelligent" --height=150 --width=250
# python cal_re_rank_img.py -d=vehicleid -a="ResNet50ta_bt10" --sampling="intelligent" --height=180 --width=360

