import os.path as osp 
import numpy as np
import math 
import torch 
import scipy
import scipy.io

smooth = 50 
alpha = 5
lamb = 0.3

def gaussian_func2(x, u, o=50):
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - u, 2)) / (2 * np.power(o, 2))
    return temp1 * np.exp(temp2)


def gauss_smooth2(arr,o):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i
    # gaussian_vect= gaussian_func2(vect,0,1)
    # o=50
    approximate_delta = 3*o     #  when x-u>approximate_delta, e.g., 6*o, the gaussian value is approximately equal to 0.
    gaussian_vect= gaussian_func2(vect,0,o)
    matrix = np.zeros((hist_num,hist_num))
    for i in range(hist_num):
        k=0
        for j in range(i,hist_num):
            if k>approximate_delta:
                continue
            matrix[i][j]=gaussian_vect[j-i] 
            k=k+1  
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2
    # for i in range(hist_num):
    #     for j in range(i):
    #         matrix[i][j]=gaussian_vect[j]     
    xxx = np.dot(matrix,arr)
    return xxx


def fliplr(img, use_gpu=True):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    if use_gpu:
        inv_idx  = inv_idx.cuda()
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def process_chunk(chunk, index):
    for i in range(0,chunk.shape[0]):
        print(index, i)
        for j in range(0,chunk.shape[1]):
            chunk[i][j][:]=gauss_smooth2(chunk[i][j][:],smooth)
    return chunk

def load_distribution(path, dataset):    
    # if osp.exists("/beegfs/pp1953/distribution_original_" + dataset + ".mat"):
    #     result2 = scipy.io.loadmat(path)
    #     distribution = result2['distribution']
    #     return distribution
    # path = "/beegfs/pp1953/distribution_vric.mat"
    result2 = scipy.io.loadmat(path)
    distribution = result2['distribution']
    #############################################################
    # import multiprocessing as mp
    # n_proc = mp.cpu_count()
    # n_proc = 8 
    # if distribution.shape[0] >  50:
    #     chunksize = distribution.shape[0] // n_proc
    #     proc_chunks = []
    #     for i_proc in range(n_proc):
    #         chunkstart = i_proc * chunksize
    #         chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else distribution.shape[0]
    #         print(chunkstart , chunkend)
    #         # make sure to include the division remainder for the last process
    #         proc_chunks.append(distribution[slice(chunkstart, chunkend)])
    #     with mp.Pool(processes=n_proc) as pool:
    #         # starts the sub-processes without blocking
    #         # pass the chunk to each worker process
    #         proc_results = [pool.apply_async(process_chunk, args=(chunk,i)) for i,chunk in enumerate(proc_chunks)]
    #         # print(proc_results)
    #         result_chunks = [r.get() for r in proc_results]
    #         distribution = np.concatenate(result_chunks, axis=0)
            
    # else:
    for i in range(0,distribution.shape[0]):
        # print(i , "\r")
        for j in range(0,distribution.shape[1]):
            # print(i,j, end="\r")
            # print("gauss "+str(i)+"->"+str(j))
            # gauss_smooth(distribution[i][j])
            distribution[i][j][:]=gauss_smooth2(distribution[i][j][:],smooth)
    eps = 0.0000001
    sum_ = np.sum(distribution,axis=2)
    for i in range(distribution.shape[0]):
        for j in range(distribution.shape[1]):
            # print(i,j, end='\r')
            distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)     
            # result = {'distribution_original':distribution}
            # scipy.io.savemat('/beegfs/pp1953/distribution_original_vric.mat',result)
    result = {'distribution':distribution}
    scipy.io.savemat("/beegfs/pp1953/distribution_original_" + dataset + ".mat" , result)

    return distribution


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc



def scoring (qf,ql,qc,qfr,gf,gl,gc,gfr,distribution):
    query = qf
    score = np.dot(gf,query)

    # spatial temporal scores: qfr,gfr, qc, gc
    # TODO
    interval = 100
    score_st = np.zeros(len(gc))
    # import pdb
    # pdb.set_trace()
    for i in range(len(gc)):
        if qfr>gfr[i]:
            diff = qfr-gfr[i]
            hist_ = int(diff/interval)
            pr = distribution[qc-1][gc[i]-1][hist_]
        else:
            diff = gfr[i]-qfr
            hist_ = int(diff/interval)
            pr = distribution[gc[i]-1][qc-1][hist_]
        score_st[i] = pr

    # ========================
    score  = 1/(1+np.exp(-alpha*score))*1/(1+2*np.exp(-alpha*score_st))

    return score
    


def evaluate(ql,qc,gl,gc,score, reverse=False):
    if reverse:    
        index = np.argsort(score)
    else:
        index = np.argsort(-score)  #from large to small

    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def eval(g_pids , q_pids , qf,  q_camids, q_frames, gf , g_camids ,g_frames ,distribution= None , **kwargs):        
        # len(g_pids), len(q_pids), qf.shape , gf.shape
        # len(q_camids) , len(q_frames) , len(g_camids) ,  len(g_frames)
        
        CMC = torch.IntTensor(len(g_pids)).zero_()
        ap = 0.0
        if type(distribution) == type(None) :
            distribution = None
        
        if "cam_dis_query" in kwargs:
            cam_q = kwargs["cam_dis_query"].astype(int)
            cam_g = kwargs["cam_dis_gallery"].astype(int) 
        else:
            cam_q = q_camids
            cam_g = g_camids
        for i in range(len(q_pids)):
            score  = scoring (qf[i], q_pids[i], cam_q[i], q_frames[i], gf, g_pids, cam_g, g_frames, distribution)
            ap_tmp, CMC_tmp = evaluate(q_pids[i],q_camids[i], g_pids,g_camids,score)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
        
        CMC = CMC.float()
        CMC = CMC/len(q_pids) #average CMC

        return CMC, ap/len(q_pids)
        
def evaluate2(score,ql,qc,gl,gc):
    index = np.argsort(score)  #from small to large
    #index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def eval2(g_pids , q_pids , qf,  q_camids, q_frames, gf , g_camids ,g_frames , all_dist):        
    CMC = torch.IntTensor(len(g_pids)).zero_()
    ap = 0.0
    #re-ranking
    print('calculate initial distance')
    re_rank = re_ranking(len(q_camids), all_dist)

    for i in range(len(q_pids)):
        ap_tmp, CMC_tmp = evaluate2(re_rank[i,:],q_pids[i],q_camids[i],g_pids,g_camids)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(q_pids) #average CMC
    return CMC, ap/len(q_pids)
    


def eval_vehicleid(q_pids, g_pids, q_camids, g_camids, query_feature, gallery_feature,max_rank, pre_compute_score=None, reverse=True):
    """Evaluation with vehicleid metric
    Key: gallery contains one images for each test vehicles and the other images in test
         use as query
    """
    use_gpu = torch.cuda.is_available()
    if pre_compute_score is None:
        query_feature = torch.FloatTensor(query_feature)
        gallery_feature = torch.FloatTensor(gallery_feature)
        if use_gpu:
            query_feature = query_feature.cuda()
            gallery_feature = gallery_feature.cuda()
        score = torch.mm(query_feature, torch.transpose( gallery_feature, 0, 1))
        score = score.cpu().numpy()
    else: 
        score = pre_compute_score

    num_q, num_g = score.shape
    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
    
    if reverse:    
        indices = np.argsort(-score, axis=1)
    else:
        indices = np.argsort(score, axis=1)


    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        # remove gallery samples that have the same pid and camid with query
        '''
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid) # original remove
        '''
        remove = False  # without camid imformation remove no images in gallery
        keep = np.invert(remove)
        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP 






# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22. 
- This version accepts distance matrix instead of raw features. 
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


import numpy as np

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(query_num, all_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = all_dist

    original_dist = 2. - 2 * original_dist   #np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    print('all_dist shape:',all_dist.shape)
    print('initial_rank shape:',initial_rank.shape)

    # query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist
