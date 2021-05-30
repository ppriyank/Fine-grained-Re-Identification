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


### Cuhk01 (p=100) or (p=485)

`--split=100` and `split=485` generates random splits. Run the experiments 10 times to get different splits 
```
python Image.py -d=cuhk01 --split=100 --opt=dataset --thresold=20 --max-epoch=500 -a="ResNet50TA_BT_image" --pretrain  --evaluate --height=256 --width=150 --split=100
```

### Market 

```



python Image.py -d=cuhk01 --split=100 --opt=dataset --thresold=20 --max-epoch=500 -a="ResNet50TA_BT_image" --pretrain  --evaluate --height=256 --width=150 --split=100
```

### VeRi

`mode == 5`, the pretrained ResNet is pretrained on `VehicleID dataset`. Similarly if you are training on VehicleID dataset, we suggest using VeRI pretrained ResNet. 







## Videos

`--seq-len` : length of video clip 





`visualize_attention_heads.py` and `plot_tsne.py` are slightly outdated code to visualize attention map and centers of center loss. 