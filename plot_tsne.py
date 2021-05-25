
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import models



arch = "ResNet50ta_bt"
model = models.init_model(name=arch, num_classes=625, loss={'xent', 'htri'})
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))



name  = "ResNet50ta_bt_best_11_checkpoint_ep161.pth.tar"
# name  = "ResNet50ta_bt_best_3_checkpoint_ep91.pth.tar"
path = "/Users/ppriyank/Code/" +  name 



checkpoint = torch.load(path , map_location=torch.device('cpu') )

centers = checkpoint['centers']['centers'].numpy()
centers = centers[:625]
state_dict = {}
for key in checkpoint['state_dict']:
        state_dict[key] = checkpoint['state_dict'][key]


model.load_state_dict(state_dict,  strict=True)

import random 
y = np.array([i for i in range(625)])
count = 625
l = random.sample(range(0,625), count)
y = y[l] 
centers = centers[l]
classifier =  model.classifier.classifier.weight.detach().numpy()[l]



feat_cols = [ 'pixel'+str(i) for i in range(centers.shape[1]) ]
df = pd.DataFrame(centers,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
rndperm = np.random.permutation(df.shape[0])

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"], 
    ys=df.loc[rndperm,:]["pca-two"], 
    zs=df.loc[rndperm,:]["pca-three"], 
    c=df.loc[rndperm,:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()




feat_cols2 = [ 'pixel'+str(i) for i in range(classifier.shape[1]) ]
df2 = pd.DataFrame(classifier,columns=feat_cols2)
df2['y'] = y
df2['label'] = df2['y'].apply(lambda i: str(i))
rndperm = np.random.permutation(df2.shape[0])

pca2 = PCA(n_components=3)
pca_result2 = pca2.fit_transform(df2[feat_cols2].values)
df2['pca-one'] = pca_result2[:,0]
df2['pca-two'] = pca_result2[:,1] 
df2['pca-three'] = pca_result2[:,2]
print('Explained variation per principal component: {}'.format(pca2.explained_variance_ratio_))

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df2.loc[rndperm,:]["pca-one"], 
    ys=df2.loc[rndperm,:]["pca-two"], 
    zs=df2.loc[rndperm,:]["pca-three"], 
    c=df2.loc[rndperm,:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()
