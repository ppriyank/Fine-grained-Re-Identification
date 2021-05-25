
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np 
import tools.data_manager as data_manager
from tools.image_eval import eval, load_distribution , fliplr
from tools.eval_metrics import evaluate , re_ranking
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools.video_loader import ImageDataset , Image_inderase, VideoDataset_inderase

dataset_name=  "vric"
dataset_name=  "msmt17"
dataset_name=  "vehicleid"
import tools.data_manager as data_manager
dataset_name=  "vehicleid_subset"
dataset = data_manager.init_dataset(name=dataset_name)

import tools.data_manager as data_manager
dataset_name=  "ilidsvid"
dataset = data_manager.init_dataset(name=dataset_name, split_id=0)

import tools.data_manager as data_manager
dataset_name=  "market2"
dataset = data_manager.init_dataset(name=dataset_name)


import tools.data_manager as data_manager
for dataset in ['mars' , 'ilidsvid', 'prid', 'cuhk01', 'market' , 'veri']:
    d = data_manager.init_dataset(name=dataset)


import tools.data_manager as data_manager
dataset = 'prid'
d = data_manager.init_dataset(name=dataset, split_id=0)


import tools.data_manager as data_manager
dataset = 'cuhk01'
d = data_manager.init_dataset(name=dataset)

import tools.data_manager as data_manager
dataset = 'ilidsvid'
d = data_manager.init_dataset(name=dataset)

import torchvision.transforms as transforms
from tools.video_loader import ImageDataset , Image_inderase, VideoDataset_inderase
from torch.utils.data import DataLoader
from tools.samplers import RandomIdentitySampler
import torch 




import scipy.io
import tools.data_manager as data_manager
import numpy as np 
from tools.eval_metrics import evaluate
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools.video_loader import VideoDataset 
import os 
import random 
from shutil import copytree , copyfile




storage_dir = "/beegfs/pp1953/"

d = data_manager.init_dataset(name="mars")
result = scipy.io.loadmat(storage_dir + "mars" +'_result.msat')
gf = result['gallery_f']
qf = result['query_f']
g_pids = result['gallery_label']
g_camids = result['gallery_cam']

q_pids = result['query_label']
q_camids = result['query_cam']
g_pids = g_pids[0]
g_camids = g_camids[0]
q_pids = q_pids[0]
q_camids = q_camids[0]

# gf = torch.tensor(gf)
# qf = torch.tensor(qf)

gf=gf.transpose()/np.power(np.sum(np.power(gf,2),axis=1),0.5)
gf=gf.transpose()
qf=qf.transpose()/np.power(np.sum(np.power(qf,2),axis=1),0.5)
qf=qf.transpose()
score = -np.matmul(qf , gf.transpose())
# distmat_rerank = re_ranking(qf2,gf2)
# cmc, mAP = evaluate(score, q_pids, g_pids, q_camids, g_camids)
# print("Results ---------- ")
# ranks=[1, 5, 10, 20]
# print("mAP: {:.1%}".format(mAP))
# print("CMC curve")
# for r in ranks:
#     print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1] ))


print("------------------")
trans = transforms.ToPILImage()
indices = np.argsort(score)
transform_test = transforms.Compose([
    transforms.Resize((250, 150), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



queryloader = DataLoader(
    VideoDataset(d.query, seq_len=4, sample='test', transform=transform_test, max_length=40),
    batch_size=1, shuffle=False, num_workers=4,
    pin_memory=True, drop_last=False,
)


galleryloader = DataLoader(
    VideoDataset(d.gallery, seq_len=4, sample='test', transform=transform_test , max_length=40),
    batch_size=1, shuffle=False, num_workers=4,
    pin_memory=True, drop_last=False,
)



def save_image(input, path):
    image = trans((input - input.min()) / (input.max() - input.min()))
    image.save(path)
        
# os.makedirs(storage_dir + "pictures/query/") 
# os.makedirs(storage_dir + "pictures/gallery/") 


  
for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
    if batch_idx == 300:
        imgs = imgs.view(-1, 3, 250, 150)
        folder = storage_dir + "pictures/" + str(batch_idx)
        try: 
            os.makedirs(folder) 
            print("Directory '%s' created successfully" %folder) 
        except OSError as error: 
            print("Directory '%s' can not be created" %(folder)) 
        for i,j in enumerate(range(imgs.shape[0])):
            path = folder + "/" + str(i) +".png"
            path
            save_image(imgs[j], path)
                




for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
    if batch_idx == 4267 or batch_idx ==  4289 : 
        imgs = imgs.view(-1, 3, 250, 150)
        folder = storage_dir + "pictures/" + str(batch_idx)
        # folder = storage_dir + "pictures/gallery/" + str(batch_idx)
        try: 
            os.makedirs(folder) 
            print("Directory '%s' created successfully" %folder) 
        except OSError as error: 
            print("Directory '%s' can not be created" %(folder)) 
        # index = random.choices(range(0,imgs.shape[0]-1), k=4)
        for i,j in enumerate(range(imgs.shape[0])):
            path = folder + "/" + str(i) +".png"
            path
            save_image(imgs[j], path)
        

        
            
num_q, num_g = score.shape
indices = np.argsort(score, axis=1)
matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
dict = {}
for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        if orig_cmc[0] == 0:
            q_idx,  order[0]
            folder = storage_dir + "pictures/query/" + str(q_idx)
            dest_folder = storage_dir + "pictures/mistake/" + str(q_idx) +"/" + str(q_idx)
            copytree(folder, dest_folder)
            folder = storage_dir + "pictures/gallery/" + str(order[0])
            dest_folder = storage_dir + "pictures/mistake/" + str(q_idx) +"/" + str(order[0])
            copytree(folder, dest_folder)
            if order[0] not in dict:
                dict[order[0]] = 0 
            dict[order[0]] +=1
            


            

num_q, num_g = score.shape
indices = np.argsort(score, axis=1)
matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
dict = {}


index = [217,218,263,264,265,266]

dest = storage_dir + "pictures/" + "white-blue"
try: 
    os.makedirs(dest) 
    print("Directory '%s' created successfully" %dest) 
except OSError as error: 
    print("Directory '%s' can not be created" %(dest)) 
    

# index = random.choices(range(0,score.shape[0]-1), k=5)
# i=1441
# index =[510, 
for i in index:
    q_idx = i
    q_pid = q_pids[q_idx]
    q_camid = q_camids[q_idx]
    order = indices[q_idx]
    remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
    keep = np.invert(remove)
    orig_cmc = matches[q_idx][keep]
    scoring = -score[q_idx][order][keep]
    order = order[keep] 
    # if not np.any(orig_cmc):
    #     continue
    q_idx, scoring[:5] , order[:5] , orig_cmc[:5]
    folder = storage_dir + "pictures/query/" + str(q_idx)
    for j in range(4):
        file1 = folder + "/" + str(j) + ".png"
        copyfile(file1 , dest + "/" + str(q_idx) + "_" + str(j) + ".png" )
    folder = storage_dir + "pictures/gallery/" 
    for k in order[:1]:
        for j in range(4):
            file2 = folder + str(k) + "/" +str(j) + ".png"
            copyfile(file2 , dest + "/" + str(q_idx) + "_" +str(k)+"_" +str(j) + ".png" )
        

        
# score = np.matmul(qf , gf.transpose())        
# (510, array([0.91664606, 0.8208056 , 0.79679364, 0.6489537 , 0.639318  ],
#       dtype=float32), array([5217, 5215, 5214, 8617, 8618]), array([1, 1, 1, 0, 0], dtype=int32))
# (1517, array([0.7783916, 0.7567166, 0.5752997, 0.5661744, 0.556179 ],
#       dtype=float32), array([8183, 8182, 5290, 5294, 5293]), array([1, 1, 0, 0, 0], dtype=int32))
# (1660, array([0.6570305 , 0.588554  , 0.5853375 , 0.56889886, 0.54494995],
#       dtype=float32), array([ 713,  249, 1959,  262,  579]), array([0, 0, 0, 0, 0], dtype=int32))
# (927, array([0.8951497 , 0.84724206, 0.8272874 , 0.6166516 , 0.5963916 ],
#       dtype=float32), array([6441, 2444, 6442, 4267, 4287]), array([1, 0, 1, 0, 0], dtype=int32))
# (201, array([0.8272027, 0.7464038, 0.7445836, 0.7131595, 0.7064924],
#       dtype=float32), array([3287, 3297, 3302, 3300, 3894]), array([0, 0, 0, 0, 1], dtype=int32))

# (112, array([0.5027785 , 0.50093925, 0.50028515, 0.49919492, 0.4845276 ],
#       dtype=float32), array([8354, 8598, 8595, 9150, 8415]), array([0, 0, 0, 0, 0], dtype=int32))

# (1415, array([0.810761 , 0.7625515, 0.7066017, 0.6968973, 0.6843778],
#       dtype=float32), array([7867, 7870, 7869, 2635, 7872]), array([0, 0, 0, 0, 0], dtype=int32))


# (362, array([0.7683811 , 0.7637851 , 0.75658596, 0.7366542 , 0.7319769 ],
#       dtype=float32), array([4601, 4609, 4605, 4590, 4594]), array([1, 1, 1, 1, 1], dtype=int32))
# (488, array([0.95969355, 0.93944377, 0.9310517 , 0.9297511 , 0.91133904],
#       dtype=float32), array([5172, 5166, 5167, 5168, 5164]), array([1, 1, 1, 1, 1], dtype=int32))
# (110, array([0.869513  , 0.7958568 , 0.77874595, 0.77814275, 0.72529554],
#       dtype=float32), array([3513, 3518, 3512, 3514, 3429]), array([1, 1, 1, 1, 0], dtype=int32))
# (1048, array([0.8994059 , 0.89788014, 0.89098954, 0.8671087 , 0.8573479 ],
#       dtype=float32), array([6755, 6756, 6757, 6758, 6750]), array([1, 1, 1, 1, 1], dtype=int32))
# (1208, array([0.95457745, 0.9357904 , 0.90585345, 0.89551073, 0.8881657 ],
#       dtype=float32), array([7329, 7339, 7337, 7328, 7331]), array([1, 1, 1, 1, 1], dtype=int32))



import os
import cv2
rootdir = "/Users/ppriyank/NYU/paper/pictures/correct/"
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        import pdb 
        pdb.set_trace()
        # os.path.join(subdir, file)
        try:
            frame = cv2.imread(os.path.join(subdir, file)) 
            cv2.imshow("Input", frame)
            cv2.waitKey(0)
        except Exception as e:
            continue 
        



# (300, array([0.67613435, 0.6605472 , 0.64959085, 0.6425261 , 0.62787956],
#       dtype=float32), array([4267, 4285, 4289, 4288, 4282]), array([1, 1, 1, 1, 1], dtype=int32))

# (335, array([0.90577376, 0.859354  , 0.85450596, 0.83941394, 0.81448257],
#       dtype=float32), array([4444, 4445, 4441, 4440, 2362]), array([1, 1, 1, 1, 0], dtype=int32))

# (357, array([0.88496333, 0.8375505 , 0.8257395 , 0.7610701 , 0.75011986],
#       dtype=float32), array([4576, 4582, 4584, 4583, 4586]), array([1, 1, 1, 1, 1], dtype=int32))
