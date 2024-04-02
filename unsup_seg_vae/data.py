import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import tifffile
import math

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)



class DatasetTrainSingle(torch.utils.data.Dataset):
    def __init__(self, args, stacks, how2treatnext1args, sample_stack_wherein, sample_isotropic, sample_continuous, preprocess_patch_reflect, n_replication, fix_r):
        self.behavior = how2treatnext1args #"ignore"/"reuse"/"avoid"
        if self.behavior == "ignore":
            assert n_replication == 1
        self.sample_isotropic  = sample_isotropic
        self.sample_continuous = sample_continuous
        self.preprocess_patch_reflect     = preprocess_patch_reflect
        self.preprocess_patch_standardize = args.preprocess_patch_standardize
        self.preprocess_patch_whiten      = args.preprocess_patch_whiten
        self.preprocess_patch_color_scale = args.preprocess_patch_color_scale
        self.preprocess_patch_color_shift = args.preprocess_patch_color_shift
        self.preprocess_patch_color_noise = args.preprocess_patch_color_noise
        self.preprocess_patch_color_scale_value = args.preprocess_patch_color_scale_value
        self.preprocess_patch_color_shift_value = args.preprocess_patch_color_shift_value
        self.preprocess_patch_color_noise_value = args.preprocess_patch_color_noise_value
        self.preprocess_patch_elastic       = args.preprocess_patch_elastic
        self.preprocess_patch_elastic_value = args.preprocess_patch_elastic_value
        self.size_input = args.size_input
        self.grid = generate_grid(self.size_input)
        self.stacks = stacks
        self.n_stack = len(self.stacks)
        self.n_rep = n_replication
        ##
        self.tn_train = 0 
        for i in range(self.n_stack):
            self.tn_train += args.n_train[i]
        if self.behavior == "reuse":
            self.v_input = sample_stack_wherein[:,0,:]
        else:
            self.v_input  = np.empty([self.tn_train, 3], dtype=float)
        self.s  = np.empty([self.tn_train               ], dtype=int)
        self.v  = np.empty([self.tn_train, self.n_rep, 3], dtype=float)
        self.r  = np.empty([self.tn_train, self.n_rep   ], dtype=float)
        self.n0 = np.empty([self.tn_train, self.n_rep, 3], dtype=float)
        self.n1 = np.empty([self.tn_train, self.n_rep, 3], dtype=float)
        self.n2 = np.empty([self.tn_train, self.n_rep, 3], dtype=float)
        ##
        if fix_r:
            r_min = (args.r_train_min+args.r_train_max) / 2
            r_max = (args.r_train_min+args.r_train_max) / 2
        else:
            r_min = args.r_train_min
            r_max = args.r_train_max
        margin = np.ceil(args.r_train_max*math.sqrt(3.0))
        for k in range(self.n_rep):
            c=0
            for i in range(self.n_stack):
                self.s[ c:c+args.n_train[i]] = i
                self.v[ c:c+args.n_train[i],k], \
                self.r[ c:c+args.n_train[i],k], \
                self.n0[c:c+args.n_train[i],k], \
                self.n1[c:c+args.n_train[i],k], \
                self.n2[c:c+args.n_train[i],k], \
                    = sample_func_continuous(\
                        self.stacks[i].shape,\
                        r_min,\
                        r_max,\
                        margin,\
                        args.r_nbh,\
                        args.n_train[i],\
                        self.sample_continuous,\
                        self.sample_isotropic,\
                        args.exponential,\
                        self.behavior,\
                        self.v_input[ c:c+args.n_train[i]],\
                        )
                c+=args.n_train[i]
    
    def __len__(self):
        return len(self.v)


    def __getitem__(self,index):
        i = index
        output = torch.zeros(self.n_rep , 1 , self.size_input , self.size_input , self.size_input)
        for j in range(self.n_rep):
            z, y, x = self.v[i,j,0], self.v[i,j,1], self.v[i,j,2]
            r = self.r[i,j]
            n0 = self.n0[i,j,:]
            n1 = self.n1[i,j,:]
            n2 = self.n2[i,j,:]
            if self.preprocess_patch_elastic:
                elastic_noise = torch.randn(self.grid.shape,device = self.grid.device) * self.preprocess_patch_elastic_value / (self.size_input - 1.0) / 2.0
                elastic_noise[:, 0, :, :,:] = 0.0
                elastic_noise[:,-1, :, :,:] = 0.0
                elastic_noise[:, :, 0, :,:] = 0.0
                elastic_noise[:, :,-1, :,:] = 0.0
                elastic_noise[:, :, :, 0,:] = 0.0
                elastic_noise[:, :, :,-1,:] = 0.0
                grid_old = self.grid + elastic_noise
            else:
                grid_old = self.grid
            z_min, z_max, y_min, y_max, x_min, x_max, grid_new = helper_func_00(z,y,x,r,n0,n1,n2,grid_old)
            block = torch.from_numpy(self.stacks[self.s[i]][z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]).unsqueeze(0).unsqueeze(0)
            patch = F.grid_sample(block,grid_new,align_corners=True)
            if self.preprocess_patch_reflect:
                for f in range(2,5):
                    if random.random() < 0.5:
                        patch = torch.flip(patch, [f])
            if self.preprocess_patch_standardize:
                sigma = math.pow(2.,random.uniform(-1., 1.))
                patch = (patch - torch.mean(patch)) / (max(torch.std(patch), 1.0 / self.size_input**1.5))
                if self.preprocess_patch_whiten:
                    raise NotImplementedError
            if self.preprocess_patch_color_scale:
                amp = math.pow(self.preprocess_patch_color_scale_value,random.uniform(-1., 1.))
                patch = patch * amp
            if self.preprocess_patch_color_shift:
                phase = random.uniform(-self.preprocess_patch_color_shift_value, self.preprocess_patch_color_shift_value)
                patch = patch + phase
            if self.preprocess_patch_color_noise:
                noise = (self.preprocess_patch_color_noise_value**0.5)*torch.randn_like(patch)
                patch = patch + noise
            output[j] = patch[0]
        return output




class DatasetVis(torch.utils.data.Dataset):
    def __init__(self, stack, poi_class, poi_coord, n_transformation, args):
        self.stack = stack
        self.poi_class = poi_class
        self.poi_coord = poi_coord
        self.n_poi = len(self.poi_class)
        self.v_input = self.poi_coord
        self.n_transformation = n_transformation
        #self.preprocess_patch_reflect     = preprocess_patch_reflect
        self.preprocess_patch_standardize = args.preprocess_patch_standardize
        self.preprocess_patch_whiten      = args.preprocess_patch_whiten
        self.size_input = args.size_input
        self.grid = generate_grid(self.size_input)
        margin = np.ceil(args.r_train_max*math.sqrt(3.0))
        ## original
        self.v_ori, self.r_ori, self.n0_ori, self.n1_ori, self.n2_ori, \
            = sample_func_continuous(\
                self.stack.shape,\
                args.r_test,\
                args.r_test,\
                margin,\
                0.0,\
                self.n_poi,\
                False,\
                False,\
                False,\
                'reuse',\
                self.v_input,\
                )
        ## rotation
        self.v_rot  = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        self.r_rot  = np.empty([self.n_transformation, self.n_poi   ], dtype=float)
        self.n0_rot = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        self.n1_rot = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        self.n2_rot = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        for i in range(self.n_transformation):
            self.v_rot[i], self.r_rot[i], self.n0_rot[i], self.n1_rot[i], self.n2_rot[i], \
                = sample_func_continuous(\
                    self.stack.shape,\
                    args.r_test,\
                    args.r_test,\
                    margin,\
                    0.0,\
                    self.n_poi,\
                    True,\
                    True,\
                    False,\
                    'reuse',\
                    self.v_input,\
                    )
        self.v_rot  = np.transpose(self.v_rot , (1, 0, 2))
        self.r_rot  = np.transpose(self.r_rot , (1, 0   ))
        self.n0_rot = np.transpose(self.n0_rot, (1, 0, 2))
        self.n1_rot = np.transpose(self.n1_rot, (1, 0, 2))
        self.n2_rot = np.transpose(self.n2_rot, (1, 0, 2))
        ## scale
        self.v_sca  = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        self.r_sca  = np.empty([self.n_transformation, self.n_poi   ], dtype=float)
        self.n0_sca = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        self.n1_sca = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        self.n2_sca = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        for i in range(self.n_transformation):
            self.v_sca[i], self.r_sca[i], self.n0_sca[i], self.n1_sca[i], self.n2_sca[i], \
                = sample_func_continuous(\
                    self.stack.shape,\
                    args.r_train_min,\
                    args.r_train_max,\
                    margin,\
                    0.0,\
                    self.n_poi,\
                    False,\
                    False,\
                    False,\
                    'reuse',\
                    self.v_input,\
                    )
        self.v_sca  = np.transpose(self.v_sca , (1, 0, 2))
        self.r_sca  = np.transpose(self.r_sca , (1, 0   ))
        self.n0_sca = np.transpose(self.n0_sca, (1, 0, 2))
        self.n1_sca = np.transpose(self.n1_sca, (1, 0, 2))
        self.n2_sca = np.transpose(self.n2_sca, (1, 0, 2))
        ## translation
        self.v_tra  = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        self.r_tra  = np.empty([self.n_transformation, self.n_poi   ], dtype=float)
        self.n0_tra = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        self.n1_tra = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        self.n2_tra = np.empty([self.n_transformation, self.n_poi, 3], dtype=float)
        for i in range(self.n_transformation):
            self.v_tra[i], self.r_tra[i], self.n0_tra[i], self.n1_tra[i], self.n2_tra[i], \
                = sample_func_continuous(\
                    self.stack.shape,\
                    args.r_test,\
                    args.r_test,\
                    margin,\
                    args.r_nbh,
                    self.n_poi,\
                    False,\
                    False,\
                    False,\
                    'reuse',\
                    self.v_input,\
                    )
        self.v_tra  = np.transpose(self.v_tra , (1, 0, 2))
        self.r_tra  = np.transpose(self.r_tra , (1, 0   ))
        self.n0_tra = np.transpose(self.n0_tra, (1, 0, 2))
        self.n1_tra = np.transpose(self.n1_tra, (1, 0, 2))
        self.n2_tra = np.transpose(self.n2_tra, (1, 0, 2))
        #



    def __len__(self):
        return self.n_poi

    def __getitem__(self,index):
        i = index
        # ori
        z_ori, y_ori, x_ori = self.v_ori[i,0], self.v_ori[i,1], self.v_ori[i,2]
        r_ori = self.r_ori[i]
        n0_ori = self.n0_ori[i,:]
        n1_ori = self.n1_ori[i,:]
        n2_ori = self.n2_ori[i,:]
        z_min_ori, z_max_ori, y_min_ori, y_max_ori, x_min_ori, x_max_ori, grid_new_ori = \
            helper_func_00(z_ori,y_ori,x_ori,r_ori,n0_ori,n1_ori,n2_ori,self.grid)
        block_ori = torch.from_numpy(self.stack[z_min_ori:z_max_ori+1,y_min_ori:y_max_ori+1,x_min_ori:x_max_ori+1]).unsqueeze(0).unsqueeze(0)
        patch_ori = F.grid_sample(block_ori,grid_new_ori,align_corners=True)        
        if self.preprocess_patch_standardize:
            patch_ori = (patch_ori - torch.mean(patch_ori)) / max(torch.std(patch_ori), 1.0 / self.size_input**1.5)
            if self.preprocess_patch_whiten:
                raise NotImplementedError
        patch_ori = patch_ori[0]
        # rot
        patches_rot = torch.zeros(self.n_transformation,1,self.size_input,self.size_input,self.size_input)
        for j in range(self.n_transformation):
            z_rot, y_rot, x_rot = self.v_rot[i,j,0], self.v_rot[i,j,1], self.v_rot[i,j,2]
            r_rot  = self.r_rot[i][j]
            n0_rot = self.n0_rot[i,j,:]
            n1_rot = self.n1_rot[i,j,:]
            n2_rot = self.n2_rot[i,j,:]
            z_min_rot, z_max_rot, y_min_rot, y_max_rot, x_min_rot, x_max_rot, grid_new_rot = \
                helper_func_00(z_rot,y_rot,x_rot,r_rot,n0_rot,n1_rot,n2_rot,self.grid)
            block_rot = torch.from_numpy(self.stack[z_min_rot:z_max_rot+1,y_min_rot:y_max_rot+1,x_min_rot:x_max_rot+1]).unsqueeze(0).unsqueeze(0)
            patch_rot = F.grid_sample(block_rot,grid_new_rot,align_corners=True)
            if self.preprocess_patch_standardize:
                patch_rot = (patch_rot - torch.mean(patch_rot)) / max(torch.std(patch_rot), 1.0 / self.size_input**1.5)
                if self.preprocess_patch_whiten:
                    raise NotImplementedError
            patches_rot[j] = patch_rot[0]
        # sca
        patches_sca = torch.zeros(self.n_transformation,1,self.size_input,self.size_input,self.size_input)
        for j in range(self.n_transformation):
            z_sca, y_sca, x_sca = self.v_sca[i,j,0], self.v_sca[i,j,1], self.v_sca[i,j,2]
            r_sca  = self.r_sca[i][j]
            n0_sca = self.n0_sca[i,j,:]
            n1_sca = self.n1_sca[i,j,:]
            n2_sca = self.n2_sca[i,j,:]
            z_min_sca, z_max_sca, y_min_sca, y_max_sca, x_min_sca, x_max_sca, grid_new_sca = \
                helper_func_00(z_sca,y_sca,x_sca,r_sca,n0_sca,n1_sca,n2_sca,self.grid)
            block_sca = torch.from_numpy(self.stack[z_min_sca:z_max_sca+1,y_min_sca:y_max_sca+1,x_min_sca:x_max_sca+1]).unsqueeze(0).unsqueeze(0)
            patch_sca = F.grid_sample(block_sca,grid_new_sca,align_corners=True)
            if self.preprocess_patch_standardize:
                patch_sca = (patch_sca - torch.mean(patch_sca)) / max(torch.std(patch_sca), 1.0 / self.size_input**1.5)
                if self.preprocess_patch_whiten:
                    raise NotImplementedError
            patches_sca[j] = patch_sca[0]
        # tra
        patches_tra = torch.zeros(self.n_transformation,1,self.size_input,self.size_input,self.size_input)
        for j in range(self.n_transformation):
            z_tra, y_tra, x_tra = self.v_tra[i,j,0], self.v_tra[i,j,1], self.v_tra[i,j,2]
            r_tra  = self.r_tra[i][j]
            n0_tra = self.n0_tra[i,j,:]
            n1_tra = self.n1_tra[i,j,:]
            n2_tra = self.n2_tra[i,j,:]
            z_min_tra, z_max_tra, y_min_tra, y_max_tra, x_min_tra, x_max_tra, grid_new_tra = \
                helper_func_00(z_tra,y_tra,x_tra,r_tra,n0_tra,n1_tra,n2_tra,self.grid)
            block_tra = torch.from_numpy(self.stack[z_min_tra:z_max_tra+1,y_min_tra:y_max_tra+1,x_min_tra:x_max_tra+1]).unsqueeze(0).unsqueeze(0)
            patch_tra = F.grid_sample(block_tra,grid_new_tra,align_corners=True)
            if self.preprocess_patch_standardize:
                patch_tra = (patch_tra - torch.mean(patch_tra)) / max(torch.std(patch_tra), 1.0 / self.size_input**1.5)
                if self.preprocess_patch_whiten:
                    raise NotImplementedError
            patches_tra[j] = patch_tra[0]
        #
        return self.poi_class[i], np.array([z_ori, y_ori, x_ori]), patch_ori, patches_rot, patches_sca, patches_tra




class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, stack, args):
        self.preprocess_patch_standardize = args.preprocess_patch_standardize
        self.preprocess_patch_whiten = args.preprocess_patch_whiten
        self.stack = stack
        self.size_input = args.size_input
        D, H, W = self.stack.shape
        self.grid = generate_grid(self.size_input)
        self.r = args.r_test
        r = self.r
        grid = np.indices( ( D-2*r, H-2*r, W-2*r ) )
        self.z = grid[0].flatten()+r
        self.y = grid[1].flatten()+r
        self.x = grid[2].flatten()+r
        
    def __len__(self):
        return len(self.z)

    def __getitem__(self,index):
        z = self.z[index]
        y = self.y[index]
        x = self.x[index]
        r = self.r
        patch = self.stack[z-r:z+r+1, y-r:y+r+1, x-r:x+r+1]
        patch = F.grid_sample(torch.from_numpy(patch).unsqueeze(0).unsqueeze(0),self.grid,align_corners=True)
        #
        if self.preprocess_patch_standardize:
            patch = (patch - torch.mean(patch)) / max(torch.std(patch), 1.0 / self.size_input**1.5)
            if self.preprocess_patch_whiten:
                raise NotImplementedError
        #
        return patch[0], np.array([z, y, x])



def helper_func_00(z,y,x,r,n0,n1,n2,grid_old):
    rz = r*(abs(n0[0])+abs(n1[0])+abs(n2[0]))
    ry = r*(abs(n0[1])+abs(n1[1])+abs(n2[1]))
    rx = r*(abs(n0[2])+abs(n1[2])+abs(n2[2]))
    rz_int = int(math.ceil(rz))
    ry_int = int(math.ceil(ry))
    rx_int = int(math.ceil(rx))
    z_int = int(z)
    y_int = int(y)
    x_int = int(x)
    z_min = z_int - rz_int
    z_max = z_int + rz_int
    y_min = y_int - ry_int
    y_max = y_int + ry_int
    x_min = x_int - rx_int
    x_max = x_int + rx_int
    grid_new = grid_old.clone().detach()
    grid_new[:,:,:,:,2] = (grid_old[:,:,:,:,2]*n0[0] + grid_old[:,:,:,:,1]*n1[0] + grid_old[:,:,:,:,0]*n2[0] ) * (r/math.ceil(rz))# * (rz / math.ceil(rz))
    grid_new[:,:,:,:,1] = (grid_old[:,:,:,:,2]*n0[1] + grid_old[:,:,:,:,1]*n1[1] + grid_old[:,:,:,:,0]*n2[1] ) * (r/math.ceil(ry))# * (ry / math.ceil(ry))
    grid_new[:,:,:,:,0] = (grid_old[:,:,:,:,2]*n0[2] + grid_old[:,:,:,:,1]*n1[2] + grid_old[:,:,:,:,0]*n2[2] ) * (r/math.ceil(rx))# * (rx / math.ceil(rx))
    return z_min, z_max, y_min, y_max, x_min, x_max, grid_new


def sample(dim_stack, r_min, r_max, n):
    assert len(dim_stack) == 3
    d, h, w = dim_stack
    assert 1 <= r_min
    assert r_min <= r_max
    assert 2*r_max+1 <= min(d,h,w)
    assert n > 0
    # result to be generated
    r = np.random.randint(r_min,r_max+1,size=n)
    vz = np.random.randint(r,d-r)
    vy = np.random.randint(r,h-r)
    vx = np.random.randint(r,w-r)
    v = np.stack((vz,vy,vx), axis=-1)
    return v, r


def sample_func_continuous(dim_stack, r_min, r_max, margin, r_nbh, n, continuous, isotropic, exponential, behavior, v_input):
    assert len(dim_stack) == 3
    d, h, w = dim_stack
    assert 1 <= r_min
    assert r_min <= r_max
    assert 2*r_max+1 <= min(d,h,w)
    assert n > 0
    #
    # sample r
    if exponential:
        r = np.exp2(np.random.uniform(np.log2(r_min),np.log2(r_max),size=n))
    else:
        r = np.random.uniform(r_min,r_max,size=n)
    #
    # sample n
    if continuous:
        #https://stackoverflow.com/questions/33658620/generating-two-orthogonal-vectors-that-are-orthogonal-to-a-particular-direction
        if isotropic:
            theta = np.random.uniform(0,2*math.pi,size=n)
            n0z = np.random.uniform(-1,1,size=n)
            #n0 = np.array([n0z, math.sqrt(1-n0z*n0z)*math.sin(theta), math.sqrt(1-n0z*n0z)*math.cos(theta)])
            #n0 = np.stack((n0z, math.sqrt(1-n0z*n0z)*math.sin(theta), math.sqrt(1-n0z*n0z)*math.cos(theta)),axis=-1)
            n0 = np.stack((n0z, np.sqrt(1-n0z*n0z)*np.sin(theta), np.sqrt(1-n0z*n0z)*np.cos(theta)),axis=-1)
        else:
            n0 = np.zeros((n,3),dtype=float)
            rand_direction = np.random.randint(2, size=n)
            n0[rand_direction==0,0] = -1.0
            n0[rand_direction==1,0] = 1.0
        n1 = np.random.randn(n,3)
        #n1 -= (n1*n0).sum(axis=1,keepdims=True).repeat(3, axis=1) * n0
        n1 -= (n1*n0).sum(axis=1,keepdims=True) * n0
        n1 /= np.linalg.norm(n1,axis=1,keepdims=True)
        n2 = np.cross(n0, n1)
    else:
        n0 = np.zeros((n,3),dtype=float)
        n1 = np.zeros((n,3),dtype=float)
        n2 = np.zeros((n,3),dtype=float)
        rand_direction_0 = np.random.randint(2, size=n)
        rand_direction_1 = np.random.randint(2, size=n)
        if isotropic:
            rand_axis_0 = np.random.randint(3, size=n)
            n0[np.logical_and(rand_axis_0 == 0,rand_direction_0 == 0),0] = -1.0
            n0[np.logical_and(rand_axis_0 == 0,rand_direction_0 == 1),0] = 1.0
            n0[np.logical_and(rand_axis_0 == 1,rand_direction_0 == 0),1] = -1.0
            n0[np.logical_and(rand_axis_0 == 1,rand_direction_0 == 1),1] = 1.0
            n0[np.logical_and(rand_axis_0 == 2,rand_direction_0 == 0),2] = -1.0
            n0[np.logical_and(rand_axis_0 == 2,rand_direction_0 == 1),2] = 1.0
        else:
            n0[rand_direction_0 == 0,0] = -1.0
            n0[rand_direction_0 == 1,0] = 1.0
        rand_axis_1 = np.random.randint(2, size=n)
        nonzero0 = np.nonzero(n0)[1]
        assert len(nonzero0) == n
        n1[np.logical_and.reduce((nonzero0 == 2 ,  rand_axis_1 == 0 , rand_direction_1 == 0)),0] = -1.0
        n1[np.logical_and.reduce((nonzero0 == 1 ,  rand_axis_1 == 1 , rand_direction_1 == 0)),0] = -1.0
        n1[np.logical_and.reduce((nonzero0 == 2 ,  rand_axis_1 == 0 , rand_direction_1 == 1)),0] = 1.0
        n1[np.logical_and.reduce((nonzero0 == 1 ,  rand_axis_1 == 1 , rand_direction_1 == 1)),0] = 1.0
        n1[np.logical_and.reduce((nonzero0 == 0 ,  rand_axis_1 == 0 , rand_direction_1 == 0)),1] = -1.0
        n1[np.logical_and.reduce((nonzero0 == 2 ,  rand_axis_1 == 1 , rand_direction_1 == 0)),1] = -1.0
        n1[np.logical_and.reduce((nonzero0 == 0 ,  rand_axis_1 == 0 , rand_direction_1 == 1)),1] = 1.0
        n1[np.logical_and.reduce((nonzero0 == 2 ,  rand_axis_1 == 1 , rand_direction_1 == 1)),1] = 1.0
        n1[np.logical_and.reduce((nonzero0 == 1 ,  rand_axis_1 == 0 , rand_direction_1 == 0)),2] = -1.0
        n1[np.logical_and.reduce((nonzero0 == 0 ,  rand_axis_1 == 1 , rand_direction_1 == 0)),2] = -1.0
        n1[np.logical_and.reduce((nonzero0 == 1 ,  rand_axis_1 == 0 , rand_direction_1 == 1)),2] = 1.0
        n1[np.logical_and.reduce((nonzero0 == 0 ,  rand_axis_1 == 1 , rand_direction_1 == 1)),2] = 1.0
        n2 = np.cross(n0, n1)
    #
    # sample v
    #margin = np.ceil(r_max*math.sqrt(3.0))
    if behavior == "reuse":
        assert r_nbh >= 0.0
        if r_nbh == 0.0:
            v = v_input
        else:
            dvz = np.random.uniform( -r_nbh, r_nbh, size=n)
            dvy = np.random.uniform( -r_nbh, r_nbh, size=n)
            dvx = np.random.uniform( -r_nbh, r_nbh, size=n)
            while True:
                temp0 = dvz*dvz + dvy*dvy + dvx*dvx > r_nbh*r_nbh
                temp1 = np.logical_or.reduce((
                            v_input[:,0]+dvz-margin < float(0),\
                            v_input[:,0]+dvz+margin > float(d-1),\
                            v_input[:,1]+dvy-margin < float(0),\
                            v_input[:,1]+dvy+margin > float(h-1),\
                            v_input[:,2]+dvx-margin < float(0),\
                            v_input[:,2]+dvx+margin > float(w-1),\
                            ))
                temp2 = np.logical_or(temp0,temp1)
                if not np.any(temp2):
                    break
                else:
                    n_amend = np.count_nonzero(temp2)
                    dvz_amend = np.random.uniform( -r_nbh, r_nbh , size=n_amend)
                    dvy_amend = np.random.uniform( -r_nbh, r_nbh , size=n_amend)
                    dvx_amend = np.random.uniform( -r_nbh, r_nbh , size=n_amend)
                    dvz[temp2] = dvz_amend
                    dvy[temp2] = dvy_amend
                    dvx[temp2] = dvx_amend
            dv = np.stack((dvz,dvy,dvx), axis=-1)
            v = v_input+dv
            v = np.around(v)
    elif behavior == "ignore":
        vz = np.random.uniform( float(0)+margin, float(d-1)-margin , size=n)
        vy = np.random.uniform( float(0)+margin, float(h-1)-margin , size=n)
        vx = np.random.uniform( float(0)+margin, float(w-1)-margin , size=n)
        v = np.stack((vz,vy,vx), axis=-1)
        v = np.around(v)
    elif behavior == "avoid":
        vz = np.random.uniform( float(0)+margin, float(d-1)-margin , size=n)
        vy = np.random.uniform( float(0)+margin, float(h-1)-margin , size=n)
        vx = np.random.uniform( float(0)+margin, float(w-1)-margin , size=n)
        while True:
            temp1 = (vz - v_input[:,0])**2 + (vy - v_input[:,1])**2 + (vx - v_input[:,2])**2 < (2*margin)**2
            if not np.any(temp1):
                break
            else:
                n_amend = np.count_nonzero(temp1)
                vz_amend = np.random.uniform( float(0)+margin, float(d-1)-margin , size=n_amend)
                vy_amend = np.random.uniform( float(0)+margin, float(h-1)-margin , size=n_amend)
                vx_amend = np.random.uniform( float(0)+margin, float(w-1)-margin , size=n_amend)
                vz[temp1] = vz_amend
                vy[temp1] = vy_amend
                vx[temp1] = vx_amend
        v = np.stack((vz,vy,vx), axis=-1)
        v = np.around(v)
    else:
        raise NotImplementedError
    return v,r,n0,n1,n2



def generate_grid(dim_input):
    D = torch.linspace(-1, 1, dim_input)
    H = torch.linspace(-1, 1, dim_input)
    W = torch.linspace(-1, 1, dim_input)
    meshz, meshy, meshx = torch.meshgrid((D, H, W))
    grid = torch.stack((meshx, meshy, meshz), 3)
    grid = grid.unsqueeze(0) # add batch dim
    return grid


def random_affine(img, min_scale=None, max_scale=None):
  # Takes and returns torch cuda tensors with channels 1st (1 img)
  # rot and shear params are in degrees
  # tf matrices need to be float32, returned as tensors
  # we don't do translations

  # https://github.com/pytorch/pytorch/issues/12362
  # https://stackoverflow.com/questions/42489310/matrix-inversion-3-3-python
  # -hard-coded-vs-numpy-linalg-inv

  # https://github.com/pytorch/vision/blob/master/torchvision/transforms
  # /functional.py#L623
  # RSS(a, scale, shear) = [cos(a) *scale   - sin(a + shear) * scale     0]
  #                        [ sin(a)*scale    cos(a + shear)*scale     0]
  #                        [     0                  0          1]
  # used by opencv functional _get_affine_matrix and
  # skimage.transform.AffineTransform

  assert (len(img.shape) == 4)
  scale = np.random.rand() * (max_scale - min_scale) + min_scale

  affine1_to_2 = np.zeros((4,4),dtype=np.float32)
  affine1_to_2[:3,:3] = ortho_group.rvs(3) * scale
  affine1_to_2[3,3] = 1.0

  affine2_to_1 = np.linalg.inv(affine1_to_2).astype(np.float32)

  affine1_to_2, affine2_to_1 = affine1_to_2[:3, :], affine2_to_1[:3, :]  # 2x3
  affine1_to_2, affine2_to_1 = torch.from_numpy(affine1_to_2), \
                               torch.from_numpy(affine2_to_1)

  img = perform_affine_tf(img.unsqueeze(dim=0), affine1_to_2.unsqueeze(dim=0))
  img = img.squeeze(dim=0)

  return img, affine1_to_2, affine2_to_1


def perform_affine_tf(data, tf_matrices):
  # expects 4D tensor, we preserve gradients if there are any

  n_i, k, d, h, w = data.shape
  n_i2, r, c = tf_matrices.shape
  assert (n_i == n_i2)
  assert (r == 3 and c == 4)

  grid = F.affine_grid(tf_matrices, data.shape, align_corners=True).to(data.device)  # output should be same size
  data_tf = F.grid_sample(data, grid,
                          padding_mode="zeros",align_corners=True)  # this can ONLY do bilinear

  return data_tf





