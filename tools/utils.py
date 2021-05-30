from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
import re
import torch


import torchvision.transforms as transforms 
import torch.nn.functional as F
import random 


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    matching_file = fpath.split("ep")[0].split("/")[-1]
    dir_ =  osp.dirname(fpath)
    for f in os.listdir(dir_):
        if re.search(matching_file, f):
            os.remove(os.path.join(dir_, f))            
    torch.save(state, fpath)
    
def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))



def resume_from_checkpoint(save_dir, arch , model ):
    file = arch +"_checkpoint_ep" 
    for f in os.listdir(save_dir):
        if re.search(file, f):
            start_epoch = int(f.split("ep")[1].split(".")[0])
            checkpoint = torch.load(osp.join(save_dir, f))
            state_dict = {}
            for key in checkpoint['state_dict']:
                    state_dict["module." + key] = checkpoint['state_dict'][key]
            model.load_state_dict(state_dict,  strict=True)
            return model  , start_epoch


