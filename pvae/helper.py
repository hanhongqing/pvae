import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal,RelaxedBernoulli
from unsup_seg_vae.distributions import WrappedNormal
import manifolds
from unsup_seg_vae.ops.manifold_layers import GeodesicLayer, MobiusLayer, LogZero, ExpZero
from unsup_seg_vae.manifolds import Euclidean, PoincareBall
from unsup_seg_vae.utils import get_mean_param

def make_distributions(args,qz_x_loc,qz_x_scale,qz_x_cov_mat,px_z_loc,px_z_scale):
    manifold = eval(args.manifold)(args.n_dim_latent, args.curvature)
    if args.posterior == 'WrappedNormal':
        assert qz_x_cov_mat is None
        qz_x = eval(args.posterior)(qz_x_loc, qz_x_scale, manifold)
    elif args.posterior == 'MultivariateNormal':
        assert qz_x_scale is None
        qz_x = eval(args.posterior)(qz_x_loc, covariance_matrix=qz_x_cov_mat )
    else:
        raise NotImplementedError
    px_z = eval(args.likelihood)(loc = px_z_loc, scale = px_z_scale)
    return qz_x, px_z
