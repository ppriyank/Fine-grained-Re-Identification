from __future__ import print_function, absolute_import
import os
import os.path as osp
from PIL import Image
import numpy as np
import sys
import math
import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
# random.seed(1)
import pickle


def read_image(img_path, height=224, width = 112):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            # width_o, height_o = img.size;
            # asp_rat = width_o/height_o;

            # new_rat = width/height;
            # if (new_rat == asp_rat):
            #     img = img.resize((width, height), Image.ANTIALIAS); 
            # else:
            #     height = round(width / asp_rat);
                #new_width = round(new_height * asp_rat);
            # img.size
            # img = img.resize((width, height), Image.ANTIALIAS);    
            # img.size
                # img.save("temp.png");

            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            sys. exit() 
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['random', 'dense', 'dense_subset' ,'intelligent_random']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        # if self.sample == 'restricted_random':
        #     frame_indices = range(num)
        #     chunks = 
        #     rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        #     begin_index = random.randint(0, rand_end)

        seq_len = self.seq_len
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            # print(begin_index, end_index, indices)
            if len(indices) < self.seq_len:
                indices=np.array(indices)
                indices = np.append(indices , [indices[-1] for i in range(self.seq_len - len(indices))])
            else:
                indices=np.array(indices)
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid

        elif self.sample == 'test':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            indices = []
            frame_indices = [i for i in range(num)]
            each = num//self.max_length
            if each > 0: 
                x = self.max_length * (each + 1) - num
                y = num - self.max_length * (each)
                ind1 = sorted(random.sample(frame_indices[:each* x], x) )
                ind2 = sorted(random.sample(frame_indices[each* x : ], y))
                indices = ind1 + ind2
            else:
                indices = frame_indices

            if len(indices) % self.seq_len != 0 :
                left = len(indices) % self.seq_len
                last_seq = indices[-left:]
                # while 
                for index in last_seq:
                    if len(last_seq) >= self.seq_len:
                        break
                    last_seq.append(index)
                indices = indices + last_seq
                
            
            imgs_list = [] 
            curr = -1
            for i in range(len(indices) // seq_len):
                imgs = []
                for i in range(seq_len):
                    curr +=1
                    index = indices[curr]
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)
            
            imgs_array = torch.stack(imgs_list)
            
            # import pdb
            # pdb.set_trace()
            
            return imgs_array, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            # import pdb
            # pdb.set_trace()
        
            cur_index=0
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        elif self.sample == 'dense_subset':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            

            cur_index=begin_index
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        
        elif self.sample == 'intelligent_random':
            # frame_indices = range(num)
            indices = []
            each = max(num//seq_len,1)
            for  i in range(seq_len):
                if i != seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            print(len(indices))
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))





from tools.transforms2 import RandomErasing3


class VideoDataset_inderase(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

        if self.sample == "intelligent":
            print("\nDistirbuted sampling chosen for dataloader as opposed to default sequence sampling\n")
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
        else:
            # frame_indices = range(num)
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            # print(len(indices), indices, num )
        imgs = []
        labels = []
        count  = 0 
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            # temp = 0 
            # img_erase = img
            img_erase , temp  = self.erase(img)
            if temp == 1:
                count +=1 
                if count == self.seq_len : 
                    img_erase = img 
                    temp = 0 
            img = img_erase 
            labels.append(temp)
            img = img.unsqueeze(0)
            imgs.append(img)
        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        #imgs=imgs.permute(1,0,2,3)
        return imgs, pid, camid , labels


class Image_inderase(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, transform=None , height=224, width = 112):
        self.dataset = dataset
        self.seq_len = seq_len
        self.transform = transform
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])
        self.height = height
        self.width = width
        # print(self.height, self.width)
        print("loading images of size %d x %d"%(self.height, self.width))
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print(index)
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        indices = []
        each = max(num//self.seq_len,1)
        for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )

        imgs = []
        labels = []
        count  = 0 
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            img = read_image(img_path, self.height, self.width)
            if self.transform is not None:
                img = self.transform(img)
            img_erase , temp  = self.erase(img)
            if temp == 1:
                count +=1 
                if count == self.seq_len : 
                    img_erase = img 
                    temp = 0 
            img = img_erase 
            img = img.unsqueeze(0)
            imgs.append(img)
            labels.append(pid)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.tensor(labels)
        return imgs, labels


def parse_frame_market(imgname, dict_cam_seq_max={}):
    # dict_cam_seq_max = {
    #     11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
    #     21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
    #     31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
    #     41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
    #     51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
    #     61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}
    dict_cam_seq_max = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 10: 0, 11: 72681, 12: 74546, 13: 74881, 14: 74761, 15: 74891, 16: 54346, 17: 0, 18: 0, 20: 0, 21: 163691, 22: 164727, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 30: 0, 31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 40: 0, 41: 72707, 42: 72473, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0, 50: 0, 51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 60: 0, 61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}
    fid = imgname.strip().split("_")[0]
    cam = int(imgname.strip().split("_")[1][1])
    seq = int(imgname.strip().split("_")[1][3])
    frame = int(imgname.strip().split("_")[2])
    count = imgname.strip().split("_")[-1]
    # print(id)
    # print(cam)  # 1
    # print(seq)  # 2
    # print(frame)
    re = 0
    for i in range(1, seq):
        re = re + dict_cam_seq_max[int(str(cam) + str(i))]
    re = re + frame
    new_name = str(fid) + "_c" + str(cam) + "_f" + '{:0>7}'.format(str(re)) + "_" + count
    # print(new_name)
    return new_name

def parse_frame_cuhk03(imgname, dict_cam_seq_max={}):
    # print(dict_cam_seq_max)
    cam = int(imgname.strip().split("_")[0])
    seq = int(imgname.strip().split("_")[2])
    frame = int(imgname.strip().split("_")[3].split(".")[0])
    re = 0
    for i in range(1, seq):
        re = re + dict_cam_seq_max[int(str(cam) + str(i))]
    re = re + frame
    
    # print(new_name)
    return '{:0>7}'.format(str(re))

class ImageDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = 'dense'

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40 , eval=False, mode=0, height=224, width=112):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.eval = eval
        self.mode = mode
        self.dict ={}
        self.height = height
        self.width = width
        # print(self.height, self.width)
        print("loading images of size %d x %d"%(self.height, self.width))
        if osp.exists("../dict.pickle"):
            with open('../dict.pickle', 'rb') as handle:
                self.dict = pickle.load(handle)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        if self.mode == 1: # for market dataset, parse renames the files
            img = read_image(img_paths, self.height, self.width)
            camid += 1
            if self.transform is not None:
                img = self.transform(img)
            img_paths = parse_frame_market(img_paths.split("/")[-1])
            img_paths = img_paths.split('_')[2][1:]
            return img, pid, camid , int(img_paths)

        if self.mode == 2:
            camid += 1
            img = read_image(img_paths, self.height, self.width)
            if self.transform is not None:
                img = self.transform(img)
            # print(img_paths)
            img_paths = img_paths.split(".")[0].split('_')[-1][1:]
            return img, pid, camid , int(img_paths)

        if self.mode == 3 or self.mode == 7:
            img = read_image(img_paths, self.height, self.width)
            if self.transform is not None:
                img = self.transform(img)
            # print(img_paths)
            return img, pid, camid
        if self.mode == 5:
            img = read_image(img_paths, self.height, self.width)
            camid += 1
            if self.transform is not None:
                img = self.transform(img)
                img_paths = int(img_paths.split("_")[-2])
            # print(img_paths)
            return img, pid, camid , img_paths

        if self.mode == 6:
            img = read_image(img_paths, self.height, self.width)
            if self.transform is not None:
                img = self.transform(img)
                name = img_paths.split("/")[-1]
                frame = name.split("_")[-1].split(".")[0][3:]
                
            # print(img_paths)
            return img, pid, camid , int(frame)
        # if self.mode == 4: # for market dataset, parse renames the files
        #     img = read_image(img_paths)
        #     camid += 1
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     # print(img_paths)
        #     # print("===" , self.dict)
        #     cam_dis = int(img_paths.split("/")[-1].split("_")[0])
        #     img_paths = parse_frame_cuhk03(img_paths.split("/")[-1], self.dict)

        #     return img, pid, camid , int(img_paths), cam_dis
        if self.mode == 4: # for market dataset, parse renames the files
            img = read_image(img_paths , self.height, self.width)
            camid += 1
            if self.transform is not None:
                img = self.transform(img)
            frame = img_paths.split("_")[-1].split(".")[0]
            return img, pid, camid , int(frame)
        num = len(img_paths)
        self.seq_len = min ( self.seq_len , num) 
        
        cur_index=0
        frame_indices = [i for i in range(num)]
        indices_list=[]
        while num-cur_index > self.seq_len:
            indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
            cur_index+=self.seq_len
        last_seq=frame_indices[cur_index:]
        # print(last_seq)
        for index in last_seq:
            if len(last_seq) >= self.seq_len:
                break
            last_seq.append(index)
        

        indices_list.append(last_seq)
        imgs_list=[]
        # print(indices_list , num , img_paths  )
        for indices in indices_list:
            if len(imgs_list) > self.max_length:
                break 
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path, self.height, self.width)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            imgs_list.append(imgs)
        imgs_array = torch.stack(imgs_list)
        return imgs_array, pid, camid    

