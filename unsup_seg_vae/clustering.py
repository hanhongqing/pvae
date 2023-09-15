import torch
from unsup_seg_vae.manifolds import Euclidean, PoincareBall
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import tifffile

def cluster_and_save(args, X,q,ID):
    ks = args.n_clustering_class
    path_output = args.path_output
    fraction = args.n_clustering_sample
    D,H,W,l = X.shape
    x = X[q:D-q,q:H-q,q:W-q,:]
    d,h,w,_ = x.shape
    n = d*h*w
    x_flat = x.reshape((n,l))
    for i in range(len(ks)):
        y = MiniBatchKMeans(n_clusters=ks[i], random_state=0).fit_predict(x_flat).reshape((d,h,w)).astype(np.uint8)
        tifffile.imwrite(path_output+"_"+str(ID)+"_"+"MiniBatchKMeans"+"_"+str(ks[i])+".tif", np.pad(y,q,mode='constant'))
    return






