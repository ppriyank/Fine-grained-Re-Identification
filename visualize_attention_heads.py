import os
import sys

# currentdir = os.path.dirname(os.path.realpath("/home/pp1953/code/Video-Person-ReID-master/current/conf_file_super_erase_image.py"))
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


import argparse
import configparser
import random 
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
# from tools import * 
import models
from tools.transforms2 import *
from tools.samplers import RandomIdentitySampler
from tools.video_loader import ImageDataset , Image_inderase , VideoDataset
import tools.data_manager as data_manager


from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker




print("Current File Name : ",os.path.realpath(__file__))



parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
parser.add_argument('-d', '--dataset', type=str, default='cuhk01',
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
# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-s', '--sampling', type=str, default='random', help="choose sampling for training")
parser.add_argument('--fin-dim', default=2048, type=int, help="final dim for center loss")


args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_gpu = torch.cuda.is_available()
args.gpu_devices = ",".join([str(i) for i in range(torch.cuda.device_count())])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


pin_memory = True if use_gpu else False
trans = transforms.ToPILImage()


def save_maps2(map, i , optional=None):
    map = map.detach().numpy()
    map = np.maximum(map, 0)
    cam = cv2.resize(map, ((args.width , args.height )) )
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    img_path = 'pictures/check%d.png'%(i)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    base_image = np.array( img, dtype='uint8' ) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(base_image)
    cam = cam / np.max(cam)
    cv2.imwrite("pictures/merged_%d.png"%(i), np.uint8(255 * cam))
    



def save_tight(filepath):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)



# img_path = 'map_%d.png'%(index)
def save_maps(map, i , optional=None, cmap='viridis', axis='on', alpha=0.7):
    im1 = F.upsample_bilinear(map.unsqueeze(0).unsqueeze(0), size=(args.height, args.width)).squeeze(0).squeeze(0)
    img_path = 'pictures/check%d.png'%(i)
    img= Image.open(img_path)
    base_image = np.array( img, dtype='uint8' )
    im1 = (im1 - im1.min()) / (im1.max() - im1.min())
    im1 = im1.cpu().detach().numpy()
    plt.imshow(base_image)
    plt.imshow(255 * im1, alpha=alpha, cmap=cmap)
    if optional :
        save_tight("pictures/merged__%s_%d.png"%(optional,i))
    else:
        save_tight("pictures/merged_%d.png"%(i))
    # img = trans()

    

def save_image(input, i= 0, optional=None):
    image = trans((input - input.min()) / (input.max() - input.min()))
    if optional != None:
        image.save("pictures/check_%s_%d.png"%(optional, i))
    else:
        image.save("pictures/check%d.png"%(i))


def test1(model_name, dataset, batch):
    args.arch = "_".join(model_name.split("_")[:2])
    dataset = data_manager.init_dataset(name="cuhk01")
    mode =  3
    transform_test = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(args)
    model = model_create(args.arch, dataset.num_train_pids, args.fin_dim)
    print("model created")
    queryloader = DataLoader(
    ImageDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test,eval=True, mode=mode, height=args.height, width=args.width),
    batch_size=batch * args.seq_len , shuffle=False, num_workers=12,
    pin_memory=pin_memory, drop_last=False,
    )
    loader = iter(queryloader)
    x = loader.next()
    x = x[0]
    for i in range(0,batch * args.seq_len, 2):
        print(i)
        save_image(x[i], i )
    b = x.size(0)
    x = model.base(x)
    h_temp = x.size(2)
    w_temp =  x.size(3)
    x = model.conv1(x)
    A_gap = model.gap(x)
    A_gap = A_gap.view(b,model.middle_dim2)
    A_gap_abs = A_gap - A_gap.min(1)[0].unsqueeze(-1).expand_as(A_gap)
    scores = A_gap_abs.softmax(1)
    scores=  scores.unsqueeze(-1).unsqueeze(-1).expand_as(x)
    attention_maps = (scores * (x - torch.min(x)) ).sum(1)
    attention_maps = attention_maps.unsqueeze(1)
    print(attention_maps.shape)
    for i in range(0,batch * args.seq_len, 2):
        print(i)
        save_maps(attention_maps[i][0], i )



