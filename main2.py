import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import os
from scipy.spatial.distance import cdist
folder_name_default="Car"

def get_feature(img):
    intensity = img.sum(axis=1)
    intensity = intensity.sum(axis=0) / (255 * img.shape[0] * img.shape[1])
    return intensity

def get_list_features(folder_name=folder_name_default):
   list_features=[]
   k=0
   for file in os.listdir(folder_name):
       #Read all file in folder to get feature
       image_temp=imread(os.path.join(folder_name,file))
       feature=get_feature(image_temp)
       list_features.append(feature)
       k=k+1

   features=np.array(list_features)
   return (features)
def init_label(N,K):
   #Random label (has value from 0 to K-1) with size (1,N)
   origin_label=np.random.randint(K,size=(1,N))
   return (origin_label)
def data_visualization(features,label,K):
   ax=plt.axes(projection='3d')
   ax.scatter(features[:,0],features[:,1],features[:,2], c=features[:,2], cmap='viridis', linewidth=0.5)
   plt.show()
def init_centroids(features,K):
   #Random each row to get K center ,this function return matrix (K,features.shape[1])
   centroids=features[np.random.choice(features.shape[0],K,replace=False)]
   return centroids
def assign_label(feature,cetroids):
    #cdist will return norm 1 of each center with all row in feature,return matrix (K,features.shape[0])
    D=cdist(feature,cetroids)
    return np.argmin(D,axis=1) #find index has norm smallest in each row
def kmeans_update_centers(features, labels, K):
    #Create new centers to update (K,features.shape[1])
    centers = np.zeros((K, features.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = features[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) ==
        set([tuple(a) for a in new_centers]))

def kmeans(features, K):
    centers = [init_centroids(features,K)]
    labels = []
    it = 0
    while True:
        labels.append(assign_label(features, centers[-1]))
        new_centers = kmeans_update_centers(features, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)
features=get_list_features(folder_name_default)
K=4
(centers,labels,it)=kmeans(features,K)

print(centers[-1])