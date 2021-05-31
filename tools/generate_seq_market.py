import os 
import sys 

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np 
import scipy.io
import tools.data_manager as data_manager

storage = "/scratch/pp1953/dataset/"
# dataset_name=  "market"s
dataset_name=  "market2"
dataset = data_manager.init_dataset(name=dataset_name)

max_cam=  6
max_seq = 8 
dict = {}
for i in range(max_cam+1):
	for j in range(max_seq+1):
		dict[ 10 * i + j ] = 0 


max_camera = 0 
for ele in dataset.train:
	for path in ele[0]:
		name = path.split("/")[-1]
		cam = int(name.strip().split("_")[1][1])
		seq = int(name.strip().split("_")[1][3])
		frame = int(name.strip().split("_")[2])
		index = int(str(cam)  + str(seq))  
		if index not in dict:
			dict[ index ] = 0 
		dict[ index ] = max(dict[ index ], frame)
		max_camera = max(cam, max_camera)


for ele in dataset.query:
		path = ele[0]
		name = path.split("/")[-1]
		cam = int(name.strip().split("_")[1][1])
		seq = int(name.strip().split("_")[1][3])
		frame = int(name.strip().split("_")[2])
		index = int(str(cam)  + str(seq))  
		if index not in dict:
			dict[ index ] = 0 
		dict[ index ] = max(dict[ index ], frame)
		max_camera = max(cam, max_camera)



for ele in dataset.gallery:
		path = ele[0]
		name = path.split("/")[-1]
		cam = int(name.strip().split("_")[1][1])
		seq = int(name.strip().split("_")[1][3])
		frame = int(name.strip().split("_")[2])
		index = int(str(cam)  + str(seq))  
		if index not in dict:
			dict[ index ] = 0 
		dict[ index ] = max(dict[ index ], frame)
		max_camera = max(cam, max_camera)

# temp = {
#         11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
#         21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
#         31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
#         41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
#         51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
#         61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}

# {11: 72681, 12: 74546, 13: 74881, 14: 74761, 15: 74891, 16: 54346, 
# 21: 163691, 22: 164727, 23: 98102, 
# 31: 161708, 32: 161769, 33: 104469, 
# 41: 72707, 42: 72473, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 
# 51: 161095, 52: 161724, 53: 103487, 
# 61: 87551, 62: 131268, 63: 95817, 64: 30952,}


# for key in temp.keys():
# 	temp[key] , dict[key]



def parse_frame(imgname, dict_cam_seq_max={}):
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


def spatial_temporal_distribution(camera_id, labels, frames):
    class_num=751
    max_hist = 5000
    spatial_temporal_sum = np.zeros((class_num,8))                       
    spatial_temporal_count = np.zeros((class_num,8))
    eps = 0.0000001
    interval = 100.0
    
    for i in range(len(camera_id)):
        label_k = labels[i]                 #### not in order, done
        cam_k = camera_id[i]-1              ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]
        spatial_temporal_sum[label_k][cam_k]=spatial_temporal_sum[label_k][cam_k]+frame_k
        spatial_temporal_count[label_k][cam_k] = spatial_temporal_count[label_k][cam_k] + 1
    spatial_temporal_avg = spatial_temporal_sum/(spatial_temporal_count+eps)          # spatial_temporal_avg: 751 ids, 8cameras, center point
    
    distribution = np.zeros((8,8,max_hist))
    for i in range(class_num):
        for j in range(8-1):
            for k in range(j+1,8):
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
    for i in range(8):
        for j in range(8):
            distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)
    
    return distribution  



camera_id = dataset.camids
# 12936
labels = []
frames = []

for ele in dataset.train    :
	for path in ele[0]:
		labels.append(ele[1])
		img_paths = parse_frame(path.split("/")[-1] , dict)
		# print(img_paths)
		frames.append(int(img_paths.split('_')[2][1:]))




distribution = spatial_temporal_distribution(camera_id, labels, frames)		

result = {'distribution':distribution}
scipy.io.savemat(storage + 'distribution_'+dataset_name+'.mat',result)

