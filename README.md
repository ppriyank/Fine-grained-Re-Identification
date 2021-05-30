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

## Images 

### Cuhk01 (p=100) or (p=485)

`--split=100` and `split=485` generates random splits. Run the experiments 10 times to get different splits 


### Market or VeRi

`python Image.py ` has all the codes related to training. It evalautes after every 

```python Image.py -d=cuhk01 --split=100 --opt=dataset --thresold=20 --max-epoch=500 -a="ResNet50TA_BT_image"  ```





`visualize_attention_heads.py` and `plot_tsne.py` are slightly outdated code to visualize attention map and centers of center loss. 