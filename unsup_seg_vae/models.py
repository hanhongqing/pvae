# https://github.com/sksq96/pytorch-vae/blob/master/vae.py
# https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb#scrollTo=nGS2f2riILy9
# https://distill.pub/2016/deconv-checkerboard/

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal,RelaxedBernoulli
from unsup_seg_vae.distributions import WrappedNormal
#from distributions import WrappedNormal
import manifolds
from unsup_seg_vae.ops.manifold_layers import GeodesicLayer, MobiusLayer, LogZero, ExpZero
from unsup_seg_vae.manifolds import Euclidean, PoincareBall
from unsup_seg_vae.utils import get_mean_param


class IWAEMergeDimensions(nn.Module):
    def __init__(self):
        super(IWAEMergeDimensions, self).__init__()
    
    def forward(self,x):
        S = x.shape[0]
        B = x.shape[1]
        R = x.shape[2:]
        return x.view(S*B,*R)

class IWAESeparateDimensions(nn.Module):
    def __init__(self,S):
        super(IWAESeparateDimensions, self).__init__()
        self.S = S
    
    def forward(self,x):
        SB = x.shape[0]
        B = SB // self.S
        R = x.shape[1:]
        return x.view(self.S,B,*R)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class ScalarMultLayer(nn.Module):
    def __init__(self, c):
        super(ScalarMultLayer, self).__init__()
        self.c = c
    def forward(self, x):
        return self.c*x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, dim_side):
        super(UnFlatten, self).__init__()
        self.dim_side = dim_side
    def forward(self, input):
        return input.view(input.size(0),-1, self.dim_side, self.dim_side, self.dim_side)


def make_encoder_convolution_layers(n_channel_encoder_conv,use_bn):
    layers = []
    for i in range(len(n_channel_encoder_conv)-1):
        in_c = n_channel_encoder_conv[i]
        out_c = n_channel_encoder_conv[i+1]
        if use_bn:
            layers += [nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)]
            layers += [nn.BatchNorm3d(out_c)]
        else:
            layers += [nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=True)]
            layers += [nn.MaxPool3d(2, stride=2, return_indices=False)]
        layers += [nn.ReLU()]
    return nn.Sequential(*layers)


def make_decoder_deconvolution_layers(n_channel_decoder_deconv,output_wo_nl,use_bn):
    layers = []
    for i in range(len(n_channel_decoder_deconv)-1):
        in_c = n_channel_decoder_deconv[i]
        out_c = n_channel_decoder_deconv[i+1]
        if i == len(n_channel_decoder_deconv)-2:
            out_c = 2*out_c
        if use_bn:
            if i < len(n_channel_decoder_deconv)-2 or not output_wo_nl:
                layers += [nn.ConvTranspose3d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1, bias=False)]
                layers += [nn.BatchNorm3d(out_c)]
        else:
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            layers += [nn.ConvTranspose3d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=True)]
        if i < len(n_channel_decoder_deconv)-2:
            layers += [nn.ReLU()]
        elif not output_wo_nl:
            layers += [nn.Sigmoid()]
        else:
            pass
    return nn.Sequential(*layers)


def make_linear_layers(n_feature_linear):
    layers = []
    assert len(n_feature_linear) > 1
    for i in range(len(n_feature_linear)-1):
        layers += [nn.Linear(in_features=n_feature_linear[i], out_features=n_feature_linear[i+1])]
        layers += [nn.ReLU()]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, args, manifold):
        super(Encoder, self).__init__()
        self.size_input = args.size_input
        self.n_channel_encoder_conv = args.n_channel_encoder_conv
        self.n_dim_side = self.size_input // (2**(len(self.n_channel_encoder_conv)-1))
        self.n_dim_flat = self.n_channel_encoder_conv[-1] * self.n_dim_side**3
        self.n_feature_encoder_linear = args.n_feature_encoder_linear
        self.n_dim_latent = args.n_dim_latent
        self.manifold = manifold
        self.conv_layers = make_encoder_convolution_layers(self.n_channel_encoder_conv,args.batchnormalize)
        self.flatten   = Flatten()
        if len(self.n_feature_encoder_linear) == 0:
            self.fc_flat2loc   = nn.Linear(in_features=self.n_dim_flat, out_features=self.n_dim_latent)
            self.fc_flat2scale = nn.Linear(in_features=self.n_dim_flat, out_features=self.n_dim_latent)
        else:
            self.fc_flat2h = make_linear_layers([self.n_dim_flat]+self.n_feature_encoder_linear)
            self.fc_h2loc   = nn.Linear(in_features=self.n_feature_encoder_linear[-1], out_features=self.n_dim_latent)
            self.fc_h2scale = nn.Linear(in_features=self.n_feature_encoder_linear[-1], out_features=self.n_dim_latent)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x) # or x = x.view()
        if len(self.n_feature_encoder_linear) == 0:
            x_loc   = self.fc_flat2loc(  x)
            x_scale = self.fc_flat2scale(x)
        else:
            x = self.fc_flat2h(x)
            x_loc   = self.fc_h2loc(  x)
            x_scale = self.fc_h2scale(x)
        x_loc   = self.manifold.expmap0(x_loc)
        x_scale = F.softplus(x_scale)+1e-5
        return x_loc, x_scale, self.manifold
        

class Decoder(nn.Module):
    def __init__(self, args, manifold):
        super(Decoder, self).__init__()
        self.size_input = args.size_input
        self.n_channel_decoder_deconv = args.n_channel_decoder_deconv
        self.n_dim_side = self.size_input // (2**(len(self.n_channel_decoder_deconv)-1))
        self.n_dim_flat = self.n_channel_decoder_deconv[0]*self.n_dim_side**3
        self.n_feature_decoder_linear = args.n_feature_decoder_linear
        self.n_dim_latent = args.n_dim_latent
        #self.manifold = args.manifold
        self.manifold = manifold
        self.output_wo_nl = args.preprocess_patch_standardize
        if len(self.n_feature_decoder_linear) == 0:
            self.fc_z2flat = nn.Linear(in_features=self.n_dim_latent, out_features=self.n_dim_flat)
        else:
            self.fc_z2h    = nn.Linear(in_features=self.n_dim_latent, out_features=self.n_feature_decoder_linear[0])
            self.fc_h2flat = make_linear_layers(self.n_feature_decoder_linear+[self.n_dim_flat])
        self.unflatten   = UnFlatten(self.n_dim_side)
        self.deconv_layers = make_decoder_deconvolution_layers(self.n_channel_decoder_deconv,self.output_wo_nl,args.batchnormalize)
        #
        self.size_kernel_gyroplane = 1
        self.gyroplane_conv = nn.Sequential(
                                            GeodesicLayer(manifold.coord_dim, self.n_dim_flat, manifold),
                                            UnFlatten(self.n_dim_side),
                                            nn.AvgPool3d(self.size_kernel_gyroplane, stride=1, padding=(self.size_kernel_gyroplane-1)//2, ceil_mode=False, count_include_pad=True),
                                            ScalarMultLayer(self.size_kernel_gyroplane**3),
                                           )
        #self.iwae_merge = IWAEMergeDimensions()
        #self.iwae_separate = IWAESeparateDimensions(args.n_sample_iwae)

        
    def forward(self, x):
        #x = self.iwae_merge(x)
        if isinstance(self.manifold,Euclidean):
            if len(self.n_feature_decoder_linear) == 0:
                x = self.fc_z2flat(x)
            else:
                x = self.fc_z2h(x)
                x = self.fc_h2flat(x)
            x = self.unflatten(x) # or x = x.view()
        elif isinstance(self.manifold,PoincareBall):
            x = self.gyroplane_conv(x)
        else:
            raise NotImplementedError
        x = self.deconv_layers(x)
        x_mean = x[:,:x.shape[1]//2,:,:,:]
        x_std  = x[:,x.shape[1]//2:,:,:,:]
        x_std = F.softplus(x_std)+1e-5
        #x = self.iwae_separate(x)
        return x_mean,x_std
        
    

class CVAE(nn.Module):
    def __init__(self,args):
        super(CVAE, self).__init__()
        self.manifold = eval(args.manifold)(args.n_dim_latent, args.curvature)
        self.enc = Encoder(args,self.manifold)
        self.dec = Decoder(args,self.manifold)
        self.prior = args.prior
        self.posterior = args.posterior
        self.likelihood = args.likelihood
        self.pz = eval(args.prior)
        self.px_z = eval(args.likelihood)
        self.qz_x = eval(args.posterior)
        self._pz_loc   = nn.Parameter(torch.zeros(args.n_dim_latent), requires_grad=False)
        self._pz_scale = nn.Parameter(torch.ones(args.n_dim_latent), requires_grad=False)    
        self.n_sample_iwae = args.n_sample_iwae
        self.iwae_merge = IWAEMergeDimensions()
        self.iwae_separate = IWAESeparateDimensions(args.n_sample_iwae)
        self.likelihood_std = args.likelihood_std
        self.fix_likelihood_std = args.fix_likelihood_std
        
    @property
    def pz_params(self):
        if self.prior == 'WrappedNormal':
            return self._pz_loc, self._pz_scale, self.manifold
        elif self.prior == 'MultivariateNormal':
            return self._pz_loc, torch.diag(self._pz_scale)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        x_loc, x_scale, self.manifold = self.enc(x)
        if self.posterior == 'WrappedNormal':
            qz_x = self.qz_x(x_loc, x_scale, self.manifold)
        elif self.posterior == 'MultivariateNormal':
            qz_x = self.qz_x(x_loc, covariance_matrix = torch.diag_embed(x_scale) )
        else:
            raise NotImplementedError
        #qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([self.n_sample_iwae]))
        px_z_mean, px_z_std = self.dec(self.iwae_merge(zs) )
        px_z_mean = self.iwae_separate(px_z_mean)
        px_z_std  = self.iwae_separate(px_z_std)
        #mean = self.iwae_separate(self.dec(self.iwae_merge(zs) ) )
        #if self.px_z == RelaxedBernoulli:
        #    px_z = self.px_z(temperature = torch.tensor(1.0).to(zs.device), probs = mean ) 
        #elif self.px_z == Normal:
        if self.fix_likelihood_std:
            px_z_std = torch.tensor(self.likelihood_std).to(zs.device)
        px_z = self.px_z(loc = px_z_mean, scale = px_z_std)
        return qz_x, px_z, zs
    
    
    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            x_loc, x_scale, self.manifold = self.enc(x)
            if self.posterior == 'WrappedNormal':
                qz_x = self.qz_x(x_loc, x_scale, self.manifold)
            elif self.posterior == 'MultivariateNormal':
                qz_x = self.qz_x(x_loc, covariance_matrix=torch.diag_embed(x_scale) )
            else:
                raise NotImplementedError
            #qz_x = self.qz_x()
            zs = qz_x.rsample(torch.Size([1]))
            #px_z_params = self.iwae_separate(self.dec(self.iwae_merge(zs) ) )
            px_z_mean, px_z_std = self.dec(zs.squeeze(0))
        return px_z_mean
    
    
    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            px_z_params = self.dec(self.pz(*self.pz_params).sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))
        return mean, \
            means.view(-1, *means.size()[2:]), \
            samples.view(-1, *samples.size()[3:])
    
    



