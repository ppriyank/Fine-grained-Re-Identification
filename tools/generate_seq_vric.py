import os 
import sys 

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np 
import scipy.io
import tools.data_manager as data_manager


storage = "/scratch/pp1953/dataset/"
dataset_name=  "vric"
dataset = data_manager.init_dataset(name=dataset_name)

max_hist = 1000

def spatial_temporal_distribution(camera_id, labels, frames):
    class_num=max(labels)+1
    max_cam = max(camera_id) 
    spatial_temporal_sum = np.zeros((class_num,max_cam))                       
    spatial_temporal_count = np.zeros((class_num,max_cam))
    eps = 0.0000001
    interval = 100.0
    for i in range(len(camera_id)):
        label_k = labels[i]                 #### not in order, done
        cam_k = camera_id[i]-1              ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]
        spatial_temporal_sum[label_k][cam_k]=spatial_temporal_sum[label_k][cam_k]+frame_k
        spatial_temporal_count[label_k][cam_k] = spatial_temporal_count[label_k][cam_k] + 1
    spatial_temporal_avg = spatial_temporal_sum/(spatial_temporal_count+eps)          # spatial_temporal_avg: 751 ids, 8cameras, center point
    
    distribution = np.zeros((max_cam,max_cam,max_hist))
    for i in range(class_num):
        for j in range(max_cam-1):
            for k in range(j+1,max_cam):
                if spatial_temporal_count[i][j]==0 or spatial_temporal_count[i][k]==0:
                    continue 
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                if st_ij>st_ik:
                    diff = st_ij-st_ik
                    hist_ = int(diff/interval)
                    distribution[j][k][hist_] = distribution[j][k][hist_]+1     # [big][small]
                else:
                    diff = st_ik-st_ij
                    hist_ = int(diff/interval)
                    distribution[k][j][hist_] = distribution[k][j][hist_]+1
    
    sum_ = np.sum(distribution,axis=2)
    for i in range(max_cam):
        for j in range(max_cam):
            distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)
    
    return distribution  


# cams = set([ele[2] for ele in dataset.query])
# len(cams)
camera_id = dataset.camids
# 12936
labels = []
frames = []

for ele in dataset.train    :
	for path in ele[0]:
		labels.append(ele[1])
		name = path.split("/")[-1]
		frame = name.split("_")[-1].split(".")[0][3:]
		frames.append(int(frame))




distribution = spatial_temporal_distribution(camera_id, labels, frames)		

result = {'distribution':distribution}
scipy.io.savemat(storage + 'distribution_'+dataset_name+'.mat',result)

