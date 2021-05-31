# Fine-grained-Re-Identification
Code for our paper Fine-Grained Re-Identification  : https://arxiv.org/pdf/2011.13475.pdf

# Citation
If you like our work, please consider citing us: 

```
@article{Pathak2020FineGrainedR,
  title={Fine-Grained Re-Identification},
  author={P. Pathak},
  journal={ArXiv},
  year={2020},
  volume={abs/2011.13475}
}
```
# Note
For concerns regarding privacy, we are not releasing the hyperparameters for our model. Consider running the hyper parameter optimization : https://github.com/facebook/Ax/blob/master/README.md

# Required libraries 
The code requires python3 
```
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install jupyter jupyterhub pandas matplotlib scipy
pip install scikit-learn scikit-image Pillow
conda clean --all --yes
```
# Dataset Setup

Change the `storage_dir` in tools/data_manager.py to the root folder of storage of the dataset. 
`self.dataset_dir` in a dataset should reflect the dataset path. (We wont be releasing any datasets, contact the original authors for the datasets).

Check the dataset is properly loaded (update the `__factory`, if your dataset is not present, else use the name highlighted in `__factory` as the way of calling the dataset ): 

```
python -c "import tools.data_manager as data_manager; dataset = data_manager.init_dataset(name='cuhk01')"
```
 
# Training the model :
Run the training scripts from current folder `cd main_scripts/`

Pre-trained ResNet models are assumed to be stored in `storage_dir +"resnet/"`.   
`mode_name` stores the name (with absolute path) of the model to be loaded, if pretraining the entire model. It doesnt load the classifier (for loading the classifier, uncomment the code in the section `args.mode_name != ''`).   
`--evaluate` saves the model after every evalaution.   
`-opt` : configuration setting, we have provided only one setting in `tools/dataset_config.conf`, add the configuration there after running hyperparameter optimization there.  
`--thresold` : number of epochs after which evalaution starts.   
`--pretrain` : Loading the pre-trained ResNets. We have decided to not provide the `mars_sota.pth.tar` or any other pretrained ResNet. 
`--fin-dim` : Dimension of the final features. 
`'--rerank'` : Do a re-rank evaluation

## Images 
Creating `st-ReID (ST)` metrics for the dataset is a recommended step (except for CUHK01 and VehicleID). The logic being : Arranging images in a seqeunce with camera and frame numbers all arranged in sequential manner. We then create a histogram distribution on them and store the distribution. When a test image comes, we assign it a histogram given its camera number and frame number. (These histograms are matches as well during evaluation boosting the accuracy significantly.)


`python Image.py ` has all the codes related to training. It evalautes after every every 10 epochs after the thresold.  

`--seq-len` : Number of positive instances in a batch

`normal` is the method of evaluating normally, embedding comparison. if `normal` is `false`, we average embedding of image and its flipped mirror reflection 

### CUHK01 (p=100) or (p=485)

`--split=100` and `split=485` generates random splits. Run the experiments 10 times to get different splits 
```
python Image.py -d=cuhk01 --split=100 --opt=dataset --thresold=20 --max-epoch=500 -a="ResNet50TA_BT_image" --pretrain  --evaluate --height=256 --width=150 --split=486 --mode-name="/scratch/pp1953/resnet/trained/ResNet50TA_BT_image_cuhk01_dataset_256_150_4_32_checkpoint_ep2.pth.tar"
```

### Market 
We are using `Market-1501-v15.09.15` dataset for experiments. Since Market is a huge dataset, we evalaute st-ReID and re-rank separately. Evalauting while training will take long, so you can just save model after every epoch. 

`generate_seq_market.py` creates distribution/histogram for the dataset, saves in the distribution in the file: `/scratch/pp1953/dataset/distribution_market2.mat` (change the storage directory)

`evaluate_image` evalautes all saved model (very very slow) using all the techinques I could find out, which worked on other datasets. 

```
python Image.py -d=market2 --opt=dataset --thresold=20 --max-epoch=500 -a="ResNet50TA_BT_image" --pretrain  --height=256 --width=150 --save-dir="/scratch/pp1953/resnet/trained/Market/"

cd ../
python tools/generate_seq_market.py

cd main_scripts/
python evaluate_image.py -d='market2' -a="ResNet50TA_BT_image" --height=256 --width=150 --save-dir="/scratch/pp1953/resnet/trained/Market/"
```

### VeRi

`mode == 5`, the pretrained ResNet is pretrained on `VehicleID dataset`. Similarly if you are training on VehicleID dataset, we suggest using VeRI pretrained ResNet. 


## Videos

For `iLIDSVID` and `PRID` use `--seed` to do expierments on different splits and average the results of 10 splits. `Mars` dataset is huge for evalauting it, the max size of clips used is 32, while for other datasets is 40. 

`--seq-len` : length of video clip   
`--num-instances` : number of instances belonging to the same class (referred as --seq-len in images)

```
python Video.py -d=cuhk01 --split=100 --opt=dataset --thresold=20 --max-epoch=500 -a="ResNet50TA_BT_image" --pretrain  --evaluate --height=256 --width=150 --split=486 --mode-name="/scratch/pp1953/resnet/trained/ResNet50TA_BT_image_cuhk01_dataset_256_150_4_32_checkpoint_ep2.pth.tar"
```

### Other datasets like : (VehicleID , VRIC, CUHK03, GRID, MSMT17, DukeMTMC_VideoReID)
Should be easy to run if you understand the code. I discarded these datasets after the premilinary experiments werent promising. 





## To do : 
clear other datasets   
merge evaluate video scripts into one  
check the code for hyper parameter optimization   



`visualize_attention_heads.py` and `plot_tsne.py` are slightly outdated code to visualize attention map and centers of center loss. 