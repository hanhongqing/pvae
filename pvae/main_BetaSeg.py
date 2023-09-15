import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,currentdir)
import datetime
import timeit
import argparse
import random
import numpy as np
import tifffile
import csv
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm
import matplotlib.pyplot as plt
from cycler import cycler
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.distributions as dist

from openTSNE import TSNE
from umap import UMAP

from data import ConcatDataset, DatasetTest, DatasetVis, DatasetTrainSingle
from models import CVAE
from objectives import loss_func_ELBO, loss_func_metric
from clustering import cluster_and_save


def parse_args():

    parser = argparse.ArgumentParser(description="")

    ### paths
    parser.add_argument('--path_train', action='append')
    parser.add_argument('--path_test',  action='append')
    parser.add_argument('--path_output')
    parser.add_argument('--path_model')
    parser.add_argument('--path_representation')
    parser.add_argument('--path_loss_plot')
    parser.add_argument('--path_loss_text')
    parser.add_argument('--path_posterior_plot')
    parser.add_argument('--path_posterior_text')
    parser.add_argument('--path_suffix_mask_cell')
    parser.add_argument('--path_suffix_mask_component', action='append')
    
    ### pipeline
    parser.add_argument('--run_train', action='store_true', default=False)
    parser.add_argument('--run_train_visualize', action='store_true', default=False)
    parser.add_argument('--run_infer', action='store_true', default=False)
    parser.add_argument('--run_cluster', action='store_true', default=False)
    
    ### technical
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num_workers_test', type=int)
    
    # data
    parser.add_argument('--max_intensity', type=int)
    parser.add_argument('--n_train', nargs="*", type=int)
    parser.add_argument('--r_train_min',   type=float,default=2.0)
    parser.add_argument('--r_train_max',   type=float,default=6.0)
    parser.add_argument('--r_test',        type=int,default=4)
    parser.add_argument('--exponential', action='store_true', default=False)
    parser.add_argument('--fix_r_anchor', action='store_true', default=False)
    parser.add_argument('--r_nbh', type=float, default=2.0)
    parser.add_argument('--sample_adaptive',   type=bool, default=True)
    parser.add_argument('--sample_isotropic',  type=bool, default=True)
    parser.add_argument('--sample_continuous', type=bool, default=True)
    parser.add_argument('--preprocess_patch_reflect',     type=bool, default=True)
    parser.add_argument('--preprocess_patch_standardize', type=bool, default=True)
    parser.add_argument('--preprocess_patch_whiten',      type=bool, default=False)
    parser.add_argument('--preprocess_patch_color_scale', type=bool, default=True)
    parser.add_argument('--preprocess_patch_color_shift', type=bool, default=True)
    parser.add_argument('--preprocess_patch_color_noise', type=bool, default=True)
    parser.add_argument('--preprocess_patch_color_scale_value', type=float, default=2.0)
    parser.add_argument('--preprocess_patch_color_shift_value', type=float, default=0.25)
    parser.add_argument('--preprocess_patch_color_noise_value', type=float, default=0.01)
    parser.add_argument('--preprocess_patch_elastic', type=bool, default=True)
    parser.add_argument('--preprocess_patch_elastic_value', type=float, default=0.25)
    parser.add_argument('--batch_size_train', type=int,default=128)
    parser.add_argument('--batch_size_test',  type=int,default=128)
    parser.add_argument('--n_pos_replicate', type=int,default=4)
    
    ### optimisation
    parser.add_argument('--n_epoch', type=int,default=1)
    parser.add_argument('--beta1', type=float, default=0.9, help='first parameter of Adam (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, help='second parameter of Adam (default: 0.900)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learnign rate for optimser (default: 1e-3)')
    parser.add_argument('--kl_beta_c', type=float, default=1e-3, help='coefficient of beta-VAE (default: 1.0)')
    parser.add_argument('--kl_beta_v', type=float, default=1e-3, help='coefficient of beta-VAE (default: 1.0)')
    parser.add_argument('--kl_analytical', type=bool, default=True)
    parser.add_argument('--ELBO_objective', choices=['VAE', 'IWAE'], default='VAE')
    parser.add_argument('--metric_beta', type=float, default=1e3)
    parser.add_argument('--metric_margin',type=float, default=1.0)
    parser.add_argument('--n_sample_push',type=int,default=128)
    
    ## geometry and statistics
    parser.add_argument('--manifold',choices=['Euclidean', 'PoincareBall'],default='Euclidean')
    parser.add_argument('--curvature',type=float,default=1.0)
    parser.add_argument('--prior',choices=['WrappedNormal', 'MultivariateNormal'],default='MultivariateNormal')
    parser.add_argument('--posterior',choices=['WrappedNormal', 'MultivariateNormal'],default='MultivariateNormal')
    parser.add_argument('--likelihood',choices=['Normal'],default='Normal')
    parser.add_argument('--prior_std',type=float,default=1.0)
    parser.add_argument('--fix_likelihood_std', action='store_true', default=False)
    parser.add_argument('--likelihood_std',type=float,default=1.0)
    
    ## architecture
    parser.add_argument('--size_input', type=int,default=16)
    parser.add_argument('--n_channel_encoder_conv',   nargs="*", type=int,default=[1,128,256,512])
    parser.add_argument('--n_feature_encoder_linear', nargs="*", type=int,default=[256])
    parser.add_argument('--n_dim_latent', type=int,default=64)
    parser.add_argument('--n_dim_latent_metric', type=int,default=8)
    parser.add_argument('--n_feature_decoder_linear', nargs="*", type=int,default=[256])
    parser.add_argument('--n_channel_decoder_deconv', nargs="*", type=int,default=[512,256,128,1])
    parser.add_argument('--batchnormalize', action='store_true', default=False)
    parser.add_argument('--n_sample_iwae',type=int,default=1)
    
    # clustering
    parser.add_argument('--n_clustering_class', nargs="*", type=float, default=[4,5,6,7,8,9,10])
    parser.add_argument('--n_clustering_sample', type=int, default=50000000) 
    return parser.parse_args()


def train(args):
    img_train = [None] * len(args.path_train)
    for i in range(len(img_train)):
        img_train[i] = tifffile.imread(args.path_train[i])
        img_train[i] = img_train[i].astype('float32')/float(args.max_intensity)
    model = CVAE(args).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
    record_loss = np.zeros([args.n_epoch, sum(args.n_train)//args.batch_size_train, 4],dtype = np.float32)
    #
    for e in range(args.n_epoch):
        dataset_train_anc = DatasetTrainSingle(args, img_train, "ignore", None,                False,                 False,                 True, 1                   , args.fix_r_anchor)
        dataset_train_pos = DatasetTrainSingle(args, img_train, "reuse",  dataset_train_anc.v, args.sample_isotropic, args.sample_continuous,True, args.n_pos_replicate, False            )
        dataset_train = ConcatDataset(dataset_train_anc , dataset_train_pos)
        dataloader_train = DataLoader(dataset_train,batch_size=args.batch_size_train, shuffle=True, drop_last=True)
        for i, x in enumerate(tqdm(dataloader_train,ascii=True)):
            optimizer.zero_grad()
            x_anc,x_pos = x
            x_anc = x_anc.to(device)
            x_pos = x_pos.to(device)
            pz = model.pz(*model.pz_params)
            qz_x_anc, px_z_anc, zs_anc = model(x_anc[:,0,:,:,:,:])
            qz_x_pos = [None] * args.n_pos_replicate
            px_z_pos = [None] * args.n_pos_replicate
            zs_pos   = [None] * args.n_pos_replicate
            for r in range(args.n_pos_replicate):
                qz_x_pos[r], px_z_pos[r], zs_pos[r] = model(x_pos[:,r,:,:,:,:])
            #
            loss_ELBO_anc = loss_func_ELBO(pz, x_anc[:,0,:,:,:,:], qz_x_anc, px_z_anc, zs_anc, args)
            loss_ELBO_pos = [None] * args.n_pos_replicate
            for r in range(args.n_pos_replicate):
                loss_ELBO_pos[r] = loss_func_ELBO(pz, x_pos[:,r,:,:,:,:], qz_x_pos[r], px_z_pos[r], zs_pos[r], args)
            loss = loss_ELBO_anc + sum(loss_ELBO_pos)
            if args.metric_objective != 'nothing':
                loss_metric = loss_func_metric(qz_x_anc, qz_x_pos, args)
                loss += args.metric_beta * loss_metric
            loss.backward()
            optimizer.step()
            record_loss[e,i,0] = loss.item()
            record_loss[e,i,1] = loss_ELBO_anc.item()
            record_loss[e,i,2] = sum(loss_ELBO_pos).item()
            if args.metric_objective != 'nothing':
                record_loss[e,i,3] = loss_metric.item()
        
    
    print("saving trained model...")
    torch.save(model.state_dict(), args.path_model)
    print("saving loss record")
    np.savetxt(args.path_loss_text, record_loss.reshape(-1,4), delimiter=",")
    

def train_visualize(args):
    record_loss = np.loadtxt(args.path_loss_text,delimiter=",")
    print("plotting loss record")
    labels = ("total","ELBO_anc","ELBO_pos","contrast")
    default_cycler = (cycler(color=['k','r','g','k']) + cycler(linestyle=['-','-','-',':']))
    fig = plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(default_cycler)
    lineObjects = ax.plot(record_loss,linewidth=0.25)
    plt.legend(lineObjects, labels,bbox_to_anchor=(1.04,1),loc='upper left')
    plt.yscale("log")
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.savefig(args.path_loss_plot, bbox_inches="tight")


def infer(args):
    print("loading test data...")
    img_test = [None] * len(args.path_test)
    for i in range(len(args.path_test)):
        img_test[i] = tifffile.imread(args.path_test[i])
        img_test[i] = img_test[i].astype('float32')/float(args.max_intensity)
    dataset_test = [None] * len(args.path_test)
    dataloader_test = [None] * len(args.path_test)
    for i in range(len(args.path_test)):
        dataset_test[i] = DatasetTest(img_test[i],args)
        dataloader_test[i] = DataLoader(dataset_test[i], batch_size=args.batch_size_test, shuffle=True, num_workers = args.num_workers_test, pin_memory=True)
    print("loading trained model...")
    model = CVAE(args).to(device)
    model.load_state_dict(torch.load(args.path_model))
    print("inferring all pixels in test stacks...")
    model.eval()
    for i in range(len(args.path_test)):
        img_representation = torch.zeros((*img_test[i].shape, args.n_dim_latent_metric), dtype = torch.float32)
        with torch.no_grad():
            for j, (x, loc) in enumerate(tqdm(dataloader_test[i],ascii=True)):
                x = x.to(device)
                mu,_,_ = model.enc(x)
                img_representation[loc[:,0], loc[:,1], loc[:,2],:] = mu.detach().cpu()[:,:args.n_dim_latent_metric]
        print("saving inferred representation...")
        torch.save(img_representation,args.path_representation+"_"+str(i)+".pt")
        


def cluster(args):
    print("loading inferred representation...")
    print('clustering...')
    r = args.r_test
    for i in range(len(args.path_test)):
        img_representation = torch.load(args.path_representation+"_"+str(i)+".pt")
        cluster_and_save(args, img_representation,r,i)



if __name__ == "__main__":
    runId = datetime.datetime.now().isoformat().replace(':','_')
    
    args = parse_args()
    args.n_dim_latent > 1
    assert args.n_channel_encoder_conv[0] == args.n_channel_decoder_deconv[-1]
    if args.likelihood == "Normal":
        assert args.preprocess_patch_standardize
    else:
        raise NotImplementedError
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    print('seed', args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    assert len(args.path_train) > 0
    assert len(args.path_test) > 0
    
    if args.run_train:
        train(args)
    if args.run_train_visualize:
        train_visualize(args)
    if args.run_infer:
        infer(args)
    if args.run_cluster:
        cluster(args)

