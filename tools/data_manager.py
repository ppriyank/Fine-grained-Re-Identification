from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
# import urllib
import urllib.request
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import random
# random.seed(1)

from tools.utils import mkdir_if_missing, write_json, read_json

"""Dataset classes"""


from collections import defaultdict

# storage_dir = "/beegfs/pp1953/"
storage_dir = "/scratch/pp1953/dataset/"

class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root  = storage_dir + "MARS"
    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, min_seq_len=0, ):
        self._check_before_run()
        
        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        video = self._process_train_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)


        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        # self.train_videos = video
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))
                # if camid in video[pid] :
                #     video[pid][camid].append(img_paths)  
                # else:
                #     video[pid][camid] =  img_paths

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _process_train_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        video = defaultdict(dict)

        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)
        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"
            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                if camid in video[pid] :
                    video[pid][camid].extend(img_paths)  
                else:
                    video[pid][camid] =  img_paths
        return video 
   

class iLIDSVID(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """
    root  = storage_dir + "iLIDS"
    dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
    data_dir = osp.join(root, 'i-LIDS-VID')
    split_dir = osp.join(root, 'train-test people splits')
    split_mat_path = osp.join(split_dir, 'train_test_splits_ilidsvid.mat')
    split_path = osp.join(root, 'splits.json')
    cam_1_path = osp.join(root, 'i-LIDS-VID/sequences/cam1')
    cam_2_path = osp.join(root, 'i-LIDS-VID/sequences/cam2')

    def __init__(self, split_id=0):
        
        self._download_data()
        self._check_before_run()
        print("**********************SPLIT ID %d **********************" %(split_id))
        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _download_data(self):
        if osp.exists(self.root):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.request
        url_opener.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = int(num_total_ids/2)

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class PRID(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root  = storage_dir + "PRID"
    # root = './data/prid2011'
    dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    split_path = osp.join(root, 'splits_prid2011.json')
    cam_a_path = osp.join(root, 'multi_shot', 'cam_a')
    cam_b_path = osp.join(root, 'multi_shot', 'cam_b')

    def __init__(self, split_id=0, min_seq_len=0):
        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet








class iLIDSVID_subset(object):
    root  = storage_dir + "iLIDS"
    
    dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
    data_dir = osp.join(root, 'i-LIDS-VID')
    split_dir = osp.join(root, 'train-test people splits')
    split_mat_path = osp.join(split_dir, 'train_test_splits_ilidsvid.mat')
    split_path = osp.join(root, 'splits.json')
    cam_1_path = osp.join(root, 'i-LIDS-VID/sequences/cam1')
    cam_2_path = osp.join(root, 'i-LIDS-VID/sequences/cam2')

    def __init__(self, split_id=0, **kwargs):
        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        # query, num_query_tracklets, num_query_pids, num_imgs_query = \
        #   self._process_data(test_dirs, cam1=True, cam2=False)
        # gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
        #   self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train
         # + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        print("=> iLIDS-VID Subset loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  ------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        noise = [] 

        if 'sampling' in kwargs :
            sampling = kwargs['sampling']
        else:
            sampling = "random" 

        if sampling == "random":
            pids = random.sample(range(1, num_train_pids), num_train_pids// 10)
            self.val_query =  []
            self.val_gallery =  []
            self.train =[]
            for tpl in train:
                if tpl[1] in pids: 
                    if tpl[2]== 0:
                        self.val_query.append(tpl)
                    else:
                        self.val_gallery.append(tpl)
                else:
                    coin = random.randint(1, 4)
                    if coin == 1:
                        noise.append(tpl)
                    self.train.append(tpl)
        else:
            print("NOT A VALID SAMPLING METHOD!!!!")
            assert False


        self.val_gallery += noise 
        self.num_train_pids = num_train_pids
        
    def _download_data(self):
        if osp.exists(self.root):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.request
        url_opener.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = int(num_total_ids/2)

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet




class PRID_subset(object):
    root  = storage_dir + "PRID"
    
    # root = './data/prid2011'
    dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    split_path = osp.join(root, 'splits_prid2011.json')
    cam_a_path = osp.join(root, 'multi_shot', 'cam_a')
    cam_b_path = osp.join(root, 'multi_shot', 'cam_b')

    def __init__(self, split_id=0, min_seq_len=0 , **kwargs):
        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-Subset loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.query = query
        self.gallery = gallery

        noise = [] 

        if 'sampling' in kwargs :
            sampling = kwargs['sampling']
        else:
            sampling = "random" 
        if sampling == "random":
            pids = random.sample(range(1, num_train_pids), num_train_pids// 10)
            self.val_query =  []
            self.val_gallery =  []
            self.train =[]
            for tpl in train:
                if tpl[1] in pids: 
                    if tpl[2]== 0:
                        self.val_query.append(tpl)
                    else:
                        self.val_gallery.append(tpl)
                else:
                    coin = random.randint(1, 4)
                    if coin == 1:
                        noise.append(tpl)
                    self.train.append(tpl)
        else:
            print("NOT A VALID SAMPLING METHOD!!!!")
            assert False


        self.val_gallery += noise 

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet



class Mars_subset2(object):
    """
    MARS
    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6
    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root  = storage_dir + "MARS"
    
    train_name_path = osp.join(root, 'info/train_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    
    def __init__(self, min_seq_len=0 , **kwargs):
        self._check_before_run()
        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
    
        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        video = self._process_train_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs 
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids 
        num_total_tracklets = num_train_tracklets 

        print("=> MARS Subset 2 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        
        self.val_query = {}
        self.val_gallery = []
        self.dict = {}
        self.val =  []
        self.train =[]
        noise = [] 
        self.pids = random.sample(range(1, num_train_pids), num_train_pids// 10)
        for tpl in train:
            if tpl[1] in self.pids:
                self.val.append(tpl)
                if tpl[1] not in self.dict :
                    self.dict[tpl[1]] = [tpl[2]]
                elif tpl[2] not in self.dict[tpl[1]]:
                    self.dict[tpl[1]].append(tpl[2])
            else:
                coin = random.randint(1, 4)
                if coin == 1:
                    noise.append(tpl)
                self.train.append(tpl)
        
        for tpl in self.val:
            if tpl[1] not in self.val_query:
                    self.val_query[tpl[1]] = {} 
            if len(self.dict[tpl[1]]) == 1:
                if tpl[2] in self.val_query[tpl[1]]:
                    self.val_query[tpl[1]][tpl[2]].extend(list(tpl[0]))
                else:
                    self.val_query[tpl[1]][tpl[2]] = list(tpl[0])
            else:
                if self.dict[tpl[1]][-1]  != tpl[2]:
                    if tpl[2] in self.val_query[tpl[1]]:
                        self.val_query[tpl[1]][tpl[2]].extend(list(tpl[0]) )
                    else:
                        self.val_query[tpl[1]][tpl[2]] = list(tpl[0])
                else:
                    self.val_gallery.append(tpl)

        val_query = []
        for pid in self.val_query.keys():
            for cam_id in self.val_query[pid]:
                val_query.append((tuple(self.val_query[pid][cam_id]), pid,cam_id))

        
        self.val_query = val_query
        self.val_gallery += noise
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids 
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))
                # if camid in video[pid] :
                #     video[pid][camid].append(img_paths)  
                # else:
                #     video[pid][camid] =  img_paths

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _process_train_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        video = defaultdict(dict)

        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)
        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"
            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                if camid in video[pid] :
                    video[pid][camid].extend(img_paths)  
                else:
                    video[pid][camid] =  img_paths
        return video 







class CUHK01(object):
    """CUHK01.
    Reference:
        Li et al. Human Reidentification with Transferred Metric Learning. ACCV 2012.
    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>`_
    
    Dataset statistics:
        - identities: 971.
        - images: 3884.
        - cameras: 4.
    """
    
    # def __init__(self, splits= 486):
    def __init__(self, splits= 100):    
        dataset_dir = 'cuhk01'
        dataset_path = storage_dir
        campus_dir = dataset_path + dataset_dir
        img_paths = sorted(glob.glob(osp.join(campus_dir, '*.png')))
        img_list = []
        pid_container = set()
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = int(img_name[:4]) - 1
            camid = (int(img_name[4:7]) - 1) // 2 # result is either 0 or 1
            img_list.append((img_path, pid, camid))
            pid_container.add(pid)

        num_pids = len(pid_container)
        num_train_pids = num_pids -splits

        order = np.arange(num_pids)
        np.random.shuffle(order)
        train_idxs = order[:num_train_pids]
        train_idxs = np.sort(train_idxs)
        idx2label = {
            idx: label
            for label, idx in enumerate(train_idxs)
        }

        train = {}
        query = [] 
        gallery = []
        count_q = 0
        count_g = 0
        for img_path, pid, camid in img_list:
            if pid in train_idxs:
                if pid in train:
                    train[pid].append(img_path)
                else:
                    train[pid] = [img_path]
            elif camid == 0:
                count_q += 1
                query.append((img_path, pid, camid))
            else:
                count_g += 1
                gallery.append((img_path, pid, camid))
        count_t = 0         
        train_per = []
        for key in train.keys():
            count_t += len(train[key])
            train_per.append((train[key] , idx2label[key] , None ))

        self.train = train_per
        self.query = query
        self.gallery = gallery 
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_pids - num_train_pids
        self.num_gallery_pids = num_pids - num_train_pids

        num_total_pids = self.num_train_pids + self.num_query_pids 
        print("=> CUHK01 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(self.num_train_pids, count_t))
        print("  query    | {:5d} | {:8d}".format(self.num_query_pids, count_q))
        print("  gallery  | {:5d} | {:8d}".format(self.num_gallery_pids, count_g))
        print("  ------------------------------")
        print("  total    | {:5d} | - ".format(num_total_pids))
        print("  ------------------------------")


class DukeMTMCreID(object):
    """DukeMTMC-reID.
    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.
    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_
    
    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    def __init__(self, root='', **kwargs):
        dataset_dir = 'DukeMTMC-reID'
        self.train_dir = storage_dir + dataset_dir + "/bounding_box_train"
        self.query_dir = storage_dir + dataset_dir + "/query"
        self.gallery_dir = storage_dir + dataset_dir + "/bounding_box_test"
        required_files = [
            dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        train , num_train_pids , num_train_imgs = self.process_dir(self.train_dir, relabel=True)
        # query , pid_query = self.process_dir(self.query_dir, relabel=False)
        # gallery , pid_gallery = self.process_dir(self.gallery_dir, relabel=False)

        query, num_query_pids, num_query_imgs = self.process_dir_test(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self.process_dir_test(self.gallery_dir, relabel=False)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> DukeMTMC-reID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")


        self.train = train
        self.query = query
        self.gallery = gallery 

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        count = 0 
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = {}
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print(pid, camid)
            if pid in data:
                data[pid].append(img_path)
            else:
                data[pid] = [img_path]
            
            assert 1 <= camid <= 8
            camid -= 1 # index starts from 0
        
        info = []
        if relabel:
            for key in data.keys():
                count += len(data[key])
                info.append([data[key]  , pid2label[key] , 0])
        
        return info , len(pid_container) , count 

    def process_dir_test(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs




class Market1501(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    def __init__(self, root='data', **kwargs):
        dataset_dir = 'market1501'
        self.train_dir = storage_dir + dataset_dir + "/bounding_box_train"
        self.query_dir = storage_dir + dataset_dir + "/query"
        self.gallery_dir = storage_dir + dataset_dir + "/bounding_box_test"

        train, num_train_pids, num_train_imgs, camids = self.process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        self.camids = camids
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        count = 0 
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = {}
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print(pid, camid)
            if pid in data:
                data[pid].append((img_path, camid))
            else:
                data[pid] = [(img_path,camid)]
            
            assert 1 <= camid <= 6
            assert 0 <= pid <= 1501  
        
        camids = []
        info = []    
        for key in data.keys():
            count += len(data[key])
            temp2 = []
            for ele in data[key]:
                camids.append(ele[1]) 
                temp2.append(ele[0])
            info.append([temp2  , pid2label[key] , 0])
        return info , len(pid_container) , count , camids

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # if pid == -1: continue  # junk images are just ignored
            assert -1 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs




class Market1501_subset(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    def __init__(self, root='data', **kwargs):
        dataset_dir = 'market1501'
        self.train_dir = storage_dir + dataset_dir + "/bounding_box_train"
        img_paths = glob.glob(osp.join(self.train_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        count = 0 
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)

        test_pids = random.sample(pid_container, len(pid_container) //2 )
        for pid in test_pids:
            pid_container.remove(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        data = {}
        test =[]
        temp = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid in pid_container:
                # print(pid, camid)
                if pid in data:
                    data[pid].append((img_path,camid))
                else:
                    data[pid] = [(img_path, camid)]
                assert 1 <= camid <= 6
                assert 0 <= pid <= 1501  
                temp.append((img_path, pid, camid))
            else:
                test.append((img_path, pid, camid))
        
        camids = []
        train = []    
        for key in data.keys():
            count += len(data[key])
            temp_2 = []
            for ele in data[key]:
                temp_2.append(ele[0])
                camids.append(ele[1])
            train.append([ temp_2 , pid2label[key] , 0])
    
        num_train_pids = len(pid_container)
        num_train_imgs = count 
        gallery = test + temp
        query = test 
        num_query_pids = len(test_pids)
        num_gallery_pids = len(test_pids)
        num_query_imgs = len(test)
        num_gallery_imgs = len(test)
        
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs 

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        self.camids = camids
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))





class CUHK03(object):
    """CUHK03.
    Reference:
        Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!>`_
    
    Dataset statistics:
        - identities: 1360.
        - images: 13164.
        - cameras: 6.
        - splits: 20 (classic).
    """
    def __init__(self,split_id=0,cuhk03_labeled=False,cuhk03_classic_split=False,**kwargs):
        dataset_dir = storage_dir + 'cuhk03'
        self.dataset_dir = dataset_dir
        if not osp.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        self.data_dir = storage_dir + 'cuhk03_release'
        self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')

        self.split_classic_det_json_path = osp.join(self.dataset_dir, 'splits_classic_detected.json')
        self.split_classic_lab_json_path = osp.join(self.dataset_dir, 'splits_classic_labeled.json')

        self.split_new_det_json_path = osp.join(self.dataset_dir, 'splits_new_detected.json')
        self.split_new_lab_json_path = osp.join(self.dataset_dir, 'splits_new_labeled.json')

        self.split_new_det_mat_path = osp.join(self.data_dir, 'cuhk03_new_protocol_config_detected.mat')
        self.split_new_lab_mat_path = osp.join(self.data_dir, 'cuhk03_new_protocol_config_labeled.mat')

        required_files = [
            self.dataset_dir, self.data_dir, self.raw_mat_path,
            self.split_new_det_mat_path, self.split_new_lab_mat_path
        ]
        # self.check_before_run(required_files)
        self.preprocess_split()

        if cuhk03_labeled:
            split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
        else:
            split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(
            splits
        ), 'Condition split_id ({}) < len(splits) ({}) is false'.format(
            split_id, len(splits)
        )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        self.process (train, query , gallery)
    
    def process (self, train, query , gallery):

        pid_container = set()
        train_pids = set([ele[1] for ele in train])
        # print(train_pids)

        test_pids = set([ele[1] for ele in query])
        data = {}
        
        for ele in train:
            path , pid, camid = ele
            if pid not in data:
                data[pid] = []
            data[pid].append((path, camid))
        train = []  
        count = 0   
        camids = [] 
        for key in data.keys():
            count += len(data[key])
            temp2 = [] 
            for ele in data[key]:
                camids.append(ele[1] + 1 )
                temp2.append(ele[0])
            train.append([temp2  , key , 0])
    
        num_train_pids = len(train_pids)
        num_train_imgs = count 
        num_query_pids = len(test_pids)
        num_gallery_pids = len(test_pids)
        num_query_imgs = len(query)
        num_gallery_imgs = len(gallery)
        
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs 

        print("=> CUHK03 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        self.camids = camids
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


    # def check_before_run(self, files):
    #     for file in files:
    #         """Check if all files are available before going deeper"""
    #         if not osp.exists(file):
    #             raise RuntimeError("'{}' is not available".format(file))
        
    def preprocess_split(self):
        # This function is a bit complex and ugly, what it does is
        # 1. extract data from cuhk-03.mat and save as png images
        # 2. create 20 classic splits (Li et al. CVPR'14)
        # 3. create new split (Zhong et al. CVPR'17)
        if osp.exists(self.imgs_labeled_dir) \
           and osp.exists(self.imgs_detected_dir) \
           and osp.exists(self.split_classic_det_json_path) \
           and osp.exists(self.split_classic_lab_json_path) \
           and osp.exists(self.split_new_det_json_path) \
           and osp.exists(self.split_new_lab_json_path):
            return

        import h5py
        import imageio
        from scipy.io import loadmat

        mkdir_if_missing(self.imgs_detected_dir)
        mkdir_if_missing(self.imgs_labeled_dir)

        print(
            'Extract image data from "{}" and save as png'.format(
                self.raw_mat_path
            )
        )
        mat = h5py.File(self.raw_mat_path, 'r')

        def _deref(ref):
            return mat[ref][:].T

        def _process_images(img_refs, campid, pid, save_dir):
            img_paths = [] # Note: some persons only have images for one view
            for imgid, img_ref in enumerate(img_refs):
                img = _deref(img_ref)
                if img.size == 0 or img.ndim < 3:
                    continue # skip empty cell
                # images are saved with the following format, index-1 (ensure uniqueness)
                # campid: index of camera pair (1-5)
                # pid: index of person in 'campid'-th camera pair
                # viewid: index of view, {1, 2}
                # imgid: index of image, (1-10)
                viewid = 1 if imgid < 5 else 2
                img_name = '{:01d}_{:03d}_{:01d}_{:02d}.png'.format(
                    campid + 1, pid + 1, viewid, imgid + 1
                )
                img_path = osp.join(save_dir, img_name)
                if not osp.isfile(img_path):
                    imageio.imwrite(img_path, img)
                img_paths.append(img_path)
            return img_paths

        def _extract_img(image_type):
            print('Processing {} images ...'.format(image_type))
            meta_data = []
            imgs_dir = self.imgs_detected_dir if image_type == 'detected' else self.imgs_labeled_dir
            for campid, camp_ref in enumerate(mat[image_type][0]):
                camp = _deref(camp_ref)
                num_pids = camp.shape[0]
                for pid in range(num_pids):
                    img_paths = _process_images(
                        camp[pid, :], campid, pid, imgs_dir
                    )
                    assert len(img_paths) > 0, \
                        'campid{}-pid{} has no images'.format(campid, pid)
                    meta_data.append((campid + 1, pid + 1, img_paths))
                print(
                    '- done camera pair {} with {} identities'.format(
                        campid + 1, num_pids
                    )
                )
            return meta_data

        meta_detected = _extract_img('detected')
        meta_labeled = _extract_img('labeled')

        def _extract_classic_split(meta_data, test_split):
            train, test = [], []
            num_train_pids, num_test_pids = 0, 0
            num_train_imgs, num_test_imgs = 0, 0
            for i, (campid, pid, img_paths) in enumerate(meta_data):

                if [campid, pid] in test_split:
                    for img_path in img_paths:
                        camid = int(
                            osp.basename(img_path).split('_')[2]
                        ) - 1 # make it 0-based
                        test.append((img_path, num_test_pids, camid))
                    num_test_pids += 1
                    num_test_imgs += len(img_paths)
                else:
                    for img_path in img_paths:
                        camid = int(
                            osp.basename(img_path).split('_')[2]
                        ) - 1 # make it 0-based
                        train.append((img_path, num_train_pids, camid))
                    num_train_pids += 1
                    num_train_imgs += len(img_paths)
            return train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs

        print('Creating classic splits (# = 20) ...')
        splits_classic_det, splits_classic_lab = [], []
        for split_ref in mat['testsets'][0]:
            test_split = _deref(split_ref).tolist()

            # create split for detected images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_detected, test_split)
            splits_classic_det.append(
                {
                    'train': train,
                    'query': test,
                    'gallery': test,
                    'num_train_pids': num_train_pids,
                    'num_train_imgs': num_train_imgs,
                    'num_query_pids': num_test_pids,
                    'num_query_imgs': num_test_imgs,
                    'num_gallery_pids': num_test_pids,
                    'num_gallery_imgs': num_test_imgs
                }
            )

            # create split for labeled images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_labeled, test_split)
            splits_classic_lab.append(
                {
                    'train': train,
                    'query': test,
                    'gallery': test,
                    'num_train_pids': num_train_pids,
                    'num_train_imgs': num_train_imgs,
                    'num_query_pids': num_test_pids,
                    'num_query_imgs': num_test_imgs,
                    'num_gallery_pids': num_test_pids,
                    'num_gallery_imgs': num_test_imgs
                }
            )

        write_json(splits_classic_det, self.split_classic_det_json_path)
        write_json(splits_classic_lab, self.split_classic_lab_json_path)

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2]) - 1 # make it 0-based
                pid = pids[idx]
                if relabel:
                    pid = pid2label[pid]
                img_path = osp.join(img_dir, img_name)
                tmp_set.append((img_path, int(pid), camid))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1 # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(
                filelist, pids, pid2label, train_idxs, img_dir, relabel=True
            )
            query_info = _extract_set(
                filelist, pids, pid2label, query_idxs, img_dir, relabel=False
            )
            gallery_info = _extract_set(
                filelist,
                pids,
                pid2label,
                gallery_idxs,
                img_dir,
                relabel=False
            )
            return train_info, query_info, gallery_info

        print('Creating new split for detected images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path), self.imgs_detected_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_det_json_path)

        print('Creating new split for labeled images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path), self.imgs_labeled_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_lab_json_path)



class GRID(object):
    """GRID.
    Reference:
        Loy et al. Multi-camera activity correlation analysis. CVPR 2009.
    URL: `<http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html>`_
    
    Dataset statistics:
        - identities: 250.
        - images: 1275.
        - cameras: 8.
    """
    def __init__(self, split_id=0, **kwargs):
        self.dataset_dir = 'GRID'
        root= storage_dir

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.probe_path = osp.join(
            self.dataset_dir, 'underground_reid', 'probe'
        )
        self.gallery_path = osp.join(
            self.dataset_dir, 'underground_reid', 'gallery'
        )
        self.split_mat_path = osp.join(
            self.dataset_dir, 'underground_reid', 'features_and_partitions.mat'
        )
        self.split_path = osp.join(self.dataset_dir, 'splits.json')

        required_files = [
            self.dataset_dir, self.probe_path, self.gallery_path,
            self.split_mat_path
        ]
        
        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, '
                'but expected between 0 and {}'.format(
                    split_id,
                    len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]

        self.process (train, query , gallery)
        
        
        
    def process (self, train, query , gallery):
        
        pid_container = set()
        train_pids = set([ele[1] for ele in train])
        for i in range(max(train_pids)):
            if i not in train_pids :
                print(i)
            
        
        test_pids_gallery = set([ele[1] for ele in gallery])
        test_pids_query = set([ele[1] for ele in query])
        faulty = []
        
        for i in test_pids_gallery:
            if i not in test_pids_query:
                faulty.append(i)
        count= 0 
        for ele in gallery:
            if ele[1] in faulty:
                count +=1

        print("faulty : ", faulty, "\tcount %d"%(count) )
        data = {}
        for ele in train:
            path , pid, camid = ele
            if pid not in data:
                data[pid] = []
            data[pid].append((path, camid))
        train = []  
        count = 0   
        camids = [] 
        for key in data.keys():
            count += len(data[key])
            temp2 = [] 
            for ele in data[key]:
                camids.append(ele[1] + 1 )
                temp2.append(ele[0])
            train.append([temp2  , key , 0])
        
        num_train_pids = len(train_pids)
        num_train_imgs = count 
        num_query_pids = len(test_pids_query)
        num_query_imgs = len(query)
        
        num_gallery_pids = len(test_pids_gallery)
        num_gallery_imgs = len(gallery)
            

        num_total_pids = len(  set(list(test_pids_gallery) + list(test_pids_gallery) +  list(test_pids_query) ) )
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> GRID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        self.camids = camids
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating 10 random splits')
            split_mat = loadmat(self.split_mat_path)
            trainIdxAll = split_mat['trainIdxAll'][0] # length = 10
            probe_img_paths = sorted(
                glob.glob(osp.join(self.probe_path, '*.jpeg'))
            )
            gallery_img_paths = sorted(
                glob.glob(osp.join(self.gallery_path, '*.jpeg'))
            )

            splits = []
            for split_idx in range(10):
                train_idxs = trainIdxAll[split_idx][0][0][2][0].tolist()
                assert len(train_idxs) == 125
                idx2label = {
                    idx: label
                    for label, idx in enumerate(train_idxs)
                }

                train, query, gallery = [], [], []

                # processing probe folder
                for img_path in probe_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx], camid))
                    else:
                        query.append((img_path, img_idx, camid))

                # process gallery folder
                for img_path in gallery_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx], camid))
                    else:
                        gallery.append((img_path, img_idx, camid))

                split = {
                    'train': train,
                    'query': query,
                    'gallery': gallery,
                    'num_train_pids': 125,
                    'num_query_pids': 125,
                    'num_gallery_pids': 900
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))




class VeRi(object):
    """VeRi.
    Reference:
        Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
    URL: `<https://vehiclereid.github.io/VeRi/>`_
    Dataset statistics:
        - identities: 775.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    
    dataset_name = "veri"

    def __init__(self, **kwargs):
        root= storage_dir
        self.dataset_dir = "VeRi"
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        # self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        self.process (train, query , gallery)
        

    def process (self, train, query , gallery):
        
        
        pid_container = set()
        train_pids = set([int(ele[1]) for ele in train])
        idx2label = {idx: label for label, idx in enumerate(train_pids)}

        # for i in range(max(train_pids)):
        #     if i not in train_pids :
        #         print(i)
        data = {}
        for ele in train:
            path , pid, camid = ele
            if pid not in data:
                data[pid] = []
            data[pid].append((path, camid))
        

        train = []  
        count = 0   
        camids = [] 
        for key in data.keys():
            count += len(data[key])
            temp2 = [] 
            for ele in data[key]:
                camids.append(ele[1] + 1 )
                temp2.append(ele[0])
            train.append([temp2  , idx2label[key] , 0])
        
        num_train_imgs = count 

        test_pids_gallery = set([ele[1] for ele in gallery])
        test_pids_query = set([ele[1] for ele in query])
        faulty = []         
        for i in test_pids_gallery:
            if i not in test_pids_query:
                faulty.append(i)

        count= 0 
        for ele in gallery:
            if ele[1] in faulty:
                count +=1

        print("faulty : ", faulty, "\tcount %d"%(count) )
        
        num_train_pids = len(train_pids)
        
        num_query_pids = len(test_pids_query)
        num_query_imgs = len(query)
        
        num_gallery_pids = len(test_pids_gallery)
        num_gallery_imgs = len(gallery)
            

        num_total_pids = len(  set(list(test_pids_gallery) + list(test_pids_gallery) +  list(test_pids_query) ) )
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> VeRi loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        self.camids = camids
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids    

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 776
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            data.append((img_path, pid, camid))

        return data



class MSMT17(object):
    """MSMT17.
    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_
    
    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    def __init__(self, root='', **kwargs):
        root = storage_dir
        self.dataset_dir = 'msmt17'
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        
        has_main_dir = False
        main_dir = "MSMT17_V2"
        
        if osp.exists(osp.join(self.dataset_dir, main_dir)):
            train_dir = 'mask_train_v2'
            test_dir = 'mask_test_v2'
            has_main_dir = True
            assert has_main_dir, 'Dataset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(
            self.dataset_dir, main_dir, 'list_train.txt'
        )
        self.list_val_path = osp.join(
            self.dataset_dir, main_dir, 'list_val.txt'
        )
        self.list_query_path = osp.join(
            self.dataset_dir, main_dir, 'list_query.txt'
        )
        self.list_gallery_path = osp.join(
            self.dataset_dir, main_dir, 'list_gallery.txt'
        )

        required_files = [self.dataset_dir, self.train_dir, self.test_dir]
        
        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        # if 'combineall' in kwargs and kwargs['combineall']:
        #     train += val

        self.process(train, query , gallery)

    def process (self, train, query , gallery):
        
        pid_container = set()
        train_pids = set([int(ele[1]) for ele in train])
        idx2label = {idx: label for label, idx in enumerate(train_pids)}

        # for i in range(max(train_pids)):
        #     if i not in train_pids :
        #         print(i)
        data = {}
        for ele in train:
            path , pid, camid = ele
            if pid not in data:
                data[pid] = []
            data[pid].append((path, camid))
        

        train = []  
        count = 0   
        camids = [] 
        for key in data.keys():
            count += len(data[key])
            temp2 = [] 
            for ele in data[key]:
                camids.append(ele[1] + 1 )
                temp2.append(ele[0])
            train.append([temp2  , idx2label[key] , 0])
        
        num_train_imgs = count 

        test_pids_gallery = set([ele[1] for ele in gallery])
        test_pids_query = set([ele[1] for ele in query])
        faulty = []         
        for i in test_pids_gallery:
            if i not in test_pids_query:
                faulty.append(i)

        count= 0 
        for ele in gallery:
            if ele[1] in faulty:
                count +=1

        print("faulty : ", faulty, "\tcount %d"%(count) )
        
        num_train_pids = len(train_pids)
        
        num_query_pids = len(test_pids_query)
        num_query_imgs = len(query)
        
        num_gallery_pids = len(test_pids_gallery)
        num_gallery_imgs = len(gallery)
            
        num_total_pids = len(  set(list(test_pids_gallery) + list(test_pids_gallery) +  list(test_pids_query) ) )
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> VeRi loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        self.camids = camids
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids    
    
    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid))

        return data



class VRIC(object):
    def __init__(self, root='', **kwargs):
        root = storage_dir
        self.dataset_dir = 'VRIC'
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        
        self.list_train_path = osp.join(self.dataset_dir, 'vric_train.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'vric_probe.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'vric_gallery.txt')

        self.train_dir = osp.join(self.dataset_dir, 'train_images/')
        self.query_dir = osp.join(self.dataset_dir, 'probe_images/')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_images/')

        train = self.process_dir(self.train_dir, self.list_train_path)
        query = self.process_dir(self.query_dir, self.list_query_path)
        gallery = self.process_dir(self.gallery_dir, self.list_gallery_path)

        self.process(train, query , gallery)

    def process (self, train, query , gallery):
        
        pid_container = set()
        train_pids = set([int(ele[1]) for ele in train])
        idx2label = {idx: label for label, idx in enumerate(train_pids)}

        data = {}
        for ele in train:
            path , pid, camid = ele
            if pid not in data:
                data[pid] = []
            data[pid].append((path, camid))
        

        train = []  
        count = 0   
        camids = [] 
        for key in data.keys():
            count += len(data[key])
            temp2 = [] 
            for ele in data[key]:
                camids.append(ele[1] + 1 )
                temp2.append(ele[0])
            train.append([temp2  , idx2label[key] , 0])
        
        num_train_imgs = count 

        test_pids_gallery = set([ele[1] for ele in gallery])
        test_pids_query = set([ele[1] for ele in query])
        faulty = []         
        for i in test_pids_gallery:
            if i not in test_pids_query:
                faulty.append(i)

        count= 0 
        for ele in gallery:
            if ele[1] in faulty:
                count +=1

        print("faulty : ", faulty, "\tcount %d"%(count) )
        
        num_train_pids = len(train_pids)
        num_query_pids = len(test_pids_query)
        num_query_imgs = len(query)
        
        num_gallery_pids = len(test_pids_gallery)
        num_gallery_imgs = len(gallery)
            
        num_total_pids = len(  set(list(test_pids_gallery) + list(test_pids_gallery) +  list(test_pids_query) ) )
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> VRIC loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        self.camids = camids
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids    
    
    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []
        min_cam = 100
        max_cam = -1
        for img_idx, img_info in enumerate(lines):
            img_path, pid , camid = img_info.split(' ')
            pid = int(pid) 
            camid = int(camid) - 1
            min_cam = min(min_cam, camid)
            max_cam = max(max_cam, camid)
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid))

        print(min_cam, max_cam)
        return data




class VehicleID(object):
    """
    VehicleID
    Reference:
    @inproceedings{liu2016deep,
    title={Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles},
    author={Liu, Hongye and Tian, Yonghong and Wang, Yaowei and Pang, Lu and Huang, Tiejun},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={2167--2175},
    year={2016}}
    Dataset statistics:
    # train_list: 13164 vehicles for model training
    # test_list_800: 800 vehicles for model testing(small test set in paper
    # test_list_1600: 1600 vehicles for model testing(medium test set in paper
    # test_list_2400: 2400 vehicles for model testing(large test set in paper
    # test_list_3200: 3200 vehicles for model testing
    # test_list_6000: 6000 vehicles for model testing
    # test_list_13164: 13164 vehicles for model testing
    """
    

    def __init__(self, root='datasets', verbose=True, test_size=800, **kwargs):
        self.dataset_dir = 'VehicleID_V1.0'
        self.root = storage_dir
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, 'image')
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.train_list = osp.join(self.split_dir, 'train_list.txt')
        self.test_size = test_size

        if self.test_size == 800:
            self.test_list = osp.join(self.split_dir, 'test_list_800.txt')
        elif self.test_size == 1600:
            self.test_list = osp.join(self.split_dir, 'test_list_1600.txt')
        elif self.test_size == 2400:
            self.test_list = osp.join(self.split_dir, 'test_list_2400.txt')

        self.label = {800: "small", 1600: "medium", 2400: "large" }
        print(self.test_list)

        self.check_before_run()

        train, query, gallery = self.process_split(relabel=True)
        
        self.process(train, query , gallery)
        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def process (self, train, query , gallery):
        
        query = [(ele[0], int(ele[1]), ele[2]) for ele in query]
        gallery = [(ele[0], int(ele[1]), ele[2]) for ele in gallery]
        pid_container = set()
        train_pids = set([int(ele[1]) for ele in train])
        idx2label = {idx: label for label, idx in enumerate(train_pids)}

        data = {}
        for ele in train:
            path , pid, camid = ele
            if pid not in data:
                data[pid] = []
            data[pid].append((path, camid))
        
        train = []  
        count = 0   
        camids = [] 
        for key in data.keys():
            count += len(data[key])
            temp2 = [] 
            for ele in data[key]:
                camids.append(ele[1] )
                temp2.append(ele[0])
            train.append([temp2  , idx2label[key] , 0])
        
        num_train_imgs = count 

        test_pids_gallery = set([ele[1] for ele in gallery])
        test_pids_query = set([ele[1] for ele in query])
        faulty = []     
        for i in test_pids_query:
            if i not in test_pids_gallery:
                faulty.append(i)
        count= 0 
        
        for ele in query:
            if ele[1] in faulty:
                count +=1
        
        print("faulty : ", faulty, "\tcount %d"%(count) )
        
        num_train_pids = len(train_pids)
        num_query_pids = len(test_pids_query)
        num_query_imgs = len(query)
        
        num_gallery_pids = len(test_pids_gallery)
        num_gallery_imgs = len(gallery)
            
        num_total_pids = len(  set(list(test_pids_gallery) + list(test_pids_gallery) +  list(test_pids_query) ) )
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs


        print("=> VehicleID (%s) loaded"% (self.label[self.test_size]))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        print("  Split {:s} :: {:8d}".format(self.label[self.test_size], num_gallery_imgs + num_query_imgs))
        self.camids = camids
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids    
    



    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError('"{}" is not available'.format(self.split_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError('"{}" is not available'.format(self.train_list))
        if self.test_size not in [800, 1600, 2400]:
            raise RuntimeError('"{}" is not available'.format(self.test_size))
        if not osp.exists(self.test_list):
            raise RuntimeError('"{}" is not available'.format(self.test_list))

    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def parse_img_pids(self, nl_pairs, pid2label=None):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            if pid2label is not None:
                pid = pid2label[pid]
            camid = 1  # don't have camid information use 1 for all
            img_path = osp.join(self.img_dir, name+'.jpg')
            output.append((img_path, pid, camid))
        return output

    def process_split(self, relabel=False):
        # read train paths
        train_pid_dict = defaultdict(list)

        # 'train_list.txt' format:
        # the first number is the number of image
        # the second number is the id of vehicle
        with open(self.train_list) as f_train:
            train_data = f_train.readlines()
            for data in train_data:
                name, pid = data.split(' ')
                pid = int(pid)
                train_pid_dict[pid].append([name, pid])
        train_pids = list(train_pid_dict.keys())
        num_train_pids = len(train_pids)
        assert num_train_pids == 13164, 'There should be 13164 vehicles for training,' \
                                        ' but but got {}, please check the data'\
                                        .format(num_train_pids)
        print('num of train ids: {}'.format(num_train_pids))
        test_pid_dict = defaultdict(list)
        with open(self.test_list) as f_test:
            test_data = f_test.readlines()
            for data in test_data:
                name, pid = data.split(' ')
                test_pid_dict[pid].append([name, pid])
        test_pids = list(test_pid_dict.keys())
        num_test_pids = len(test_pids)
        assert num_test_pids == self.test_size, 'There should be {} vehicles for testing,' \
                                                ' but but got {}, please check the data'\
                                                .format(self.test_size, num_test_pids)

        train_data = []
        query_data = []
        gallery_data = []

        # for train ids, all images are used in the train set.
        for pid in train_pids:
            imginfo = train_pid_dict[pid]  # imginfo include image name and id
            train_data.extend(imginfo)

        # for each test id, random choose one image for gallery
        # and the other ones for query.
        for pid in test_pids:
            imginfo = test_pid_dict[pid]
            sample = random.choice(imginfo)
            imginfo.remove(sample)
            query_data.extend(imginfo)
            gallery_data.append(sample)

        if relabel:
            train_pid2label = self.get_pid2label(train_pids)
        else:
            train_pid2label = None
        # for key, value in train_pid2label.items():
        #     print('{key}:{value}'.format(key=key, value=value))

        train = self.parse_img_pids(train_data, train_pid2label)
        query = self.parse_img_pids(query_data)
        gallery = self.parse_img_pids(gallery_data)
        return train, query, gallery




class VehicleID_subset(object):
    
    def __init__(self, root='datasets', verbose=True, **kwargs):
        self.dataset_dir = 'VehicleID_V1.0'
        self.root = storage_dir
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, 'image')
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.train_list = osp.join(self.split_dir, 'train_list.txt')
        
        self.check_before_run()

        train = self.process_split(relabel=True)
        
        self.process(train)
        
    def process (self, train, subset_size=2000, percentage=75, discarded=9000):
        
        pid_container = set()
        train_pids = set([int(ele[1]) for ele in train])
        # print("===", len(train_pids))
        discarded_pid = random.sample(train_pids, discarded)         
        train_pids = [i for i in train_pids if i not in discarded_pid]
        test_pid = random.sample(train_pids, subset_size)         
        train_pids = [i for i in train_pids if i not in test_pid ]
        # print("===", len(train_pids))
        idx2label = {idx: label for label, idx in enumerate(train_pids)}

        data = {}
        gallery = []
        query = []
        for ele in train:
            path , pid, camid = ele
            if pid in discarded_pid:
                continue
            if pid not in data:
                data[pid] = []
            data[pid].append((path))

        
        train = []  
        count = 0   
        for key in data.keys():
            if key in train_pids:
                temp = random.sample(data[key] , max(1,int(percentage / 100 * len(data[key])) // 1) )         
                count += len(temp)
                train.append([temp  , idx2label[key] , 0])
            else:
                gallery.append( [ data[key][-1] , key, 1 ]  ) 
                query.append( [ data[key][0] , key, 1 ]  ) 
                    
        num_train_imgs = count 

        
        test_pids_gallery = set([ele[1] for ele in gallery])
        test_pids_query = set([ele[1] for ele in query])
        faulty = []     
        for i in test_pids_query:
            if i not in test_pids_gallery:
                faulty.append(i)
        count= 0 
        
        for ele in query:
            if ele[1] in faulty:
                count +=1
        
        print("faulty : ", faulty, "\tcount %d"%(count) )
        
        num_train_pids = len(train_pids)
        num_query_pids = len(test_pids_query)
        num_query_imgs = len(query)
        
        num_gallery_pids = len(test_pids_gallery)
        num_gallery_imgs = len(gallery)
            
        num_total_pids = len(  set(list(test_pids_gallery) + list(test_pids_gallery) +  list(test_pids_query) ) )
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        
        print("=> VehicleID Subset (%d) loaded"% (subset_size))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        print("  Split {:d} :: {:8d}".format(subset_size, num_gallery_imgs + num_query_imgs))
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids    

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError('"{}" is not available'.format(self.split_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError('"{}" is not available'.format(self.train_list))
        
    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def parse_img_pids(self, nl_pairs, pid2label=None):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            if pid2label is not None:
                pid = pid2label[pid]
            camid = 1  # don't have camid information use 1 for all
            img_path = osp.join(self.img_dir, name+'.jpg')
            output.append((img_path, pid, camid))
        return output

    def process_split(self, relabel=False):
        # read train paths
        train_pid_dict = defaultdict(list)

        # 'train_list.txt' format:
        # the first number is the number of image
        # the second number is the id of vehicle
        with open(self.train_list) as f_train:
            train_data = f_train.readlines()
            for data in train_data:
                name, pid = data.split(' ')
                pid = int(pid)
                train_pid_dict[pid].append([name, pid])
        train_pids = list(train_pid_dict.keys())
        num_train_pids = len(train_pids)
        assert num_train_pids == 13164, 'There should be 13164 vehicles for training,' \
                                        ' but but got {}, please check the data'\
                                        .format(num_train_pids)
        print('num of train ids: {}'.format(num_train_pids))
        
        train_data = []
        for pid in train_pids:
            imginfo = train_pid_dict[pid]  # imginfo include image name and id
            train_data.extend(imginfo)

        if relabel:
            train_pid2label = self.get_pid2label(train_pids)
        else:
            train_pid2label = None
        
        train = self.parse_img_pids(train_data, train_pid2label)
        return train


class DukeMTMC_VideoReID(object):
    """DukeMTMCVidReID.
    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target,
          Multi-Camera Tracking. ECCVW 2016.
        - Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
          Re-Identification by Stepwise Learning. CVPR 2018.
    URL: `<https://github.com/Yu-Wu/DukeMTMC-VideoReID>`_
    
    Dataset statistics:
        - identities: 702 (train) + 702 (test).
        - tracklets: 2196 (train) + 2636 (test).
    """
    
    def __init__(self, min_seq_len=0, **kwargs):
        dataset_dir = 'DukeMTMC-VideoReID'
        root = storage_dir
        dataset_dir = osp.join(root, dataset_dir)

        train_dir = osp.join(dataset_dir, 'train')
        query_dir = osp.join(dataset_dir, 'query')
        gallery_dir = osp.join(dataset_dir, 'gallery')

        split_train_json_path = osp.join(dataset_dir, 'split_train.json')
        split_query_json_path = osp.join(dataset_dir, 'split_query.json')
        split_gallery_json_path = osp.join(dataset_dir, 'split_gallery.json')


        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self.process_dir(train_dir, split_train_json_path, relabel=True)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self.process_dir(query_dir, split_query_json_path, relabel=False)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self.process_dir(gallery_dir, split_gallery_json_path, relabel=False)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> DukeMTMC_VideoReID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        # self.train_videos = video
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


    def process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']  

        print('=> Generating split json file (** this might take a while **)')
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print(
            'Processing "{}" with {} person identities'.format(
                dir_path, len(pdirs)
            )
        )

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        num_imgs_per_tracklet = []
        tracklets = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel:
                pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)
                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx + 1).zfill(4)
                    res = glob.glob(
                        osp.join(tdir, '*' + img_idx_name + '*.jpg')
                    )
                    if len(res) == 0:
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))


        num_pids = len(pdirs)
        num_tracklets = len(tracklets)
        
        print('Saving split to {}'.format(json_path))
        split_dict = {'tracklets': tracklets , 'num_pids': num_pids, 'num_imgs_per_tracklet': num_imgs_per_tracklet, 'num_tracklets': num_tracklets}
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet





class Market1501_2(object):
    """Market1501.
    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.dataset_dir = 'Market-1501-v15.09.15'
        _junk_pids = [0, -1]
        self.root = storage_dir
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = self.data_dir
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)


        self.process_dir2(train, query , gallery)
        
        

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        return data
    def process_dir2(self, train, query , gallery):
        num_imgs_per_tracklet = []
        tracklets = []
        
        data = {}
        for ele in train:
            path , pid, camid = ele
            if pid not in data:
                data[pid] = []
            data[pid].append((path, camid))

        train = []  
        count = 0   
        camids = [] 
        for key in data.keys():
            count += len(data[key])
            temp2 = [] 
            for ele in data[key]:
                camids.append(ele[1] + 1 )
                temp2.append(ele[0])
            train.append([temp2  , key  , 0])
        
        num_train_imgs = count 

        test_pids_gallery = set([ele[1] for ele in gallery])
        test_pids_query = set([ele[1] for ele in query])
        faulty = []         
        for i in test_pids_query:
            if i not in test_pids_gallery:
                faulty.append(i)

        count= 0 
        for ele in query:
            if ele[1] in faulty:
                count +=1

        print("faulty : ", faulty, "\tcount %d"%(count) )
        
        num_train_pids = len(data.keys())
        num_query_pids = len(test_pids_query)
        num_query_imgs = len(query)
        
        num_gallery_pids = len(test_pids_gallery)
        num_gallery_imgs = len(gallery)
            
        num_total_pids = len(  set(list(test_pids_gallery) + list(test_pids_gallery) +  list(test_pids_query) ) )
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
       
        print("=> MARKET  loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        self.camids = camids
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids    
    


__factory = {
    'duke_video' : DukeMTMC_VideoReID,
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid': PRID,
    'vehicleid': VehicleID,
    'vric': VRIC, 
    'cuhk01': CUHK01,
    'cuhk03': CUHK03,
    'grid' : GRID,
    'veri': VeRi,
    'msmt17': MSMT17,
    'mars_subset2' :Mars_subset2,
    'prid_subset' :PRID_subset,
    'ilidsvid_subset' : iLIDSVID_subset,

    'duke' : DukeMTMCreID,
    'market' : Market1501,
    'market2' : Market1501_2,
    'market_sub' : Market1501_subset,
    'vehicleid_subset': VehicleID_subset,
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)

