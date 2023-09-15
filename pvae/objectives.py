import torch
import torch.distributions as dist
from numpy import prod
from unsup_seg_vae.utils import has_analytic_kl, log_mean_exp
import torch.nn.functional as F

def loss_func_ELBO(pz, x, qz_x, px_z, zs, args):
    beta_c = args.kl_beta_c
    beta_v = args.kl_beta_v
    likelihood = args.likelihood
    analytical_kl = args.kl_analytical
    objective = args.ELBO_objective
    nd = args.n_dim_latent_metric
    S = zs.shape[0]
    B = zs.shape[1]
    x_expanded = x.unsqueeze(0).expand(S,*x.shape)
    # lpx_z
    lpx_z = px_z.log_prob(x_expanded).view(S,B,-1)#.mean(-1)
    # kl divergence
    if has_analytic_kl(type(qz_x), type(pz)) and analytical_kl:
        kld = dist.kl_divergence(qz_x, pz).unsqueeze(0).expand(S,B,-1)#.mean(-1) #or .sum(-1)
    else:
        lpz = pz.log_prob(zs)
        lqz_x = qz_x.log_prob(zs)
        kld = lqz_x - lpz
    # obj
    if objective=='IWAE':
        obj = -log_mean_exp(lpx_z.view(kld.squeeze(-1).shape) - kld.squeeze(-1)).sum()
    elif objective=='VAE':
        l = kld.shape[2]
        temp =  kld.mean(0).mean(0)
        obj = -lpx_z.mean().mean().mean() + (beta_c*temp[:nd].sum()+beta_v*temp[nd:].sum())/l
    else:
        raise NotImplementedError
    return obj

def kl(mu_0,var_0,mu_1,var_1):
    loss_temp = 0.5*(var_1+(mu_1-mu_0)**2)/var_0 - 0.5 + torch.log(var_0/var_1)
    loss = loss_temp.mean()
    return loss

def loss_func_metric(qz_x_anc,qz_x_pos,args):
    objective_metric = args.metric_objective
    margin = args.metric_margin
    nd = args.n_dim_latent_metric
    #shapes: (bs,dim_z) (bs,dim_z,dim_z) (bs,dim_z)
    n_rep = len(qz_x_pos)
    bs,dim_z = qz_x_anc.loc.shape
    mu_anc = qz_x_anc.loc
    cov_anc = qz_x_anc.covariance_matrix
    var_anc = torch.diagonal(cov_anc, dim1=1, dim2=2)
    mu_anc_c = mu_anc[:,:nd]
    mu_anc_v = mu_anc[:,nd:]
    var_anc_c = var_anc[:,:nd]
    var_anc_v = var_anc[:,nd:]
    mu_pos  = torch.zeros(n_rep,bs,dim_z).to(mu_anc.device)
    var_pos = torch.zeros(n_rep,bs,dim_z).to(var_anc.device)
    mu_pos_c  = torch.zeros(n_rep,bs,nd).to(mu_anc.device)
    var_pos_c = torch.zeros(n_rep,bs,nd).to(var_anc.device)
    mu_pos_v  = torch.zeros(n_rep,bs,dim_z-nd).to(mu_anc.device)
    var_pos_v = torch.zeros(n_rep,bs,dim_z-nd).to(var_anc.device)
    for r in range(n_rep):
        mu_pos[ r] = qz_x_pos[r].loc
        var_pos[r] = torch.diagonal(qz_x_pos[r].covariance_matrix, dim1=1, dim2=2)
        mu_pos_c[ r] = mu_pos[ r][:,:nd]
        var_pos_c[r] = var_pos[r][:,:nd]
        mu_pos_v[ r] = mu_pos[ r][:,nd:]
        var_pos_v[r] = var_pos[r][:,nd:]
    ##
    mu_all_c  = torch.cat((mu_anc_c.unsqueeze( 0),mu_pos_c ),0)
    temp0 = mu_all_c.unsqueeze(-2).unsqueeze(-2).expand(1+n_rep, bs, 1+n_rep, bs, nd)
    temp1 = mu_all_c.unsqueeze( 0).unsqueeze( 0).expand(1+n_rep, bs, 1+n_rep, bs, nd)
    dist = torch.norm(temp0 - temp1,dim=4)
    del temp0
    del temp1
    loss = 0.0
    alpha = 1.0
    beta  = 1.0
    for i in range(bs):
        batch_idx = torch.cat((torch.arange(0,i),torch.arange(i+1,bs))).type('torch.LongTensor')
        for a in range(1+n_rep):
            rep_idx = torch.cat((torch.arange(0,a),torch.arange(a+1,1+n_rep))).type('torch.LongTensor')
            loss += 1.0 / alpha * torch.logsumexp( alpha * (torch.cat((torch.flatten(dist[a,i,rep_idx,i        ]),torch.tensor([margin],device=dist.device))) - margin) ,dim=0)
            loss += 1.0 / beta  * torch.logsumexp( -beta * (torch.cat((torch.flatten(dist[a,i,:      ,batch_idx]),torch.tensor([margin],device=dist.device))) - margin) ,dim=0)
    loss = loss / (bs*(n_rep+1))

    return loss







