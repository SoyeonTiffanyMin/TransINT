#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:58:27 2019

@author: TiffMin
"""
import os
import math
import pickle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy, time

from projection import *
from utils import to_cuda
from model import *
torch.cuda.set_device(0)
USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor
else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor
################
#Angle Script 
################

#1. Load Model 
#Link prediction no filter dim 100 best, night tmux 1
PATH = '/data/scratch-oc40/symin95/thesis_kg_project/transint/knowledge_representation_pytorch-master/model/KALE/FB122/dim_100_hr_gap1_no_norm_ent_f0imp1/l_0.005_es_1000_L_1_em_100_nb_100_n_1000_m_5.0_f_0_mo_0.9_s_0_op_1_lo_0/TransINT.ckpt'
PATH_imp = '/data/scratch-oc40/symin95/thesis_kg_project/transint/knowledge_representation_pytorch-master/model/KALE/FB122/dim_100_hr_gap1_no_norm_ent_f1imp1/l_0.003_es_1000_L_1_em_100_nb_100_n_1000_m_5.0_f_1_mo_0.9_s_0_op_1_lo_0/TransINT.ckpt'
mod=torch.load(PATH_imp)
mod.eval()
mod = mod.cuda() 

#1-1: Pull out parameters for the P's and r's 
num_ent = mod.ent_embeddings.weight.data.shape[0]
num_rel = 122
rel_embeddings = mod.vvrel_all_weights()

unique_rel_tensor = longTensor(range(122))
unique_bases = mod.comb_mats(unique_rel_tensor) #this is AT, A is unique_bases.t()
ATA_inverse =  mod.inverse(torch.bmm(unique_bases, unique_bases.transpose(1,2)))
unique_projmat =to_cuda(torch.cat([torch.eye(mod.embedding_size)]*num_rel).view(num_rel, mod.embedding_size, mod.embedding_size)) - torch.bmm(torch.bmm(unique_bases.transpose(1,2), ATA_inverse), unique_bases).type(floatTensor) #num_rel x emb x emb
rel_embeddings_column_pytorch = to_cuda(torch.tensor(rel_embeddings)).view(num_rel,mod.embedding_size,1).type(floatTensor) #originally: embed_dim x num_ent
proj_all_r = torch.bmm(unique_projmat, rel_embeddings_column_pytorch).squeeze(2)#num_rel x emb

rel_idx = pickle.load(open('datasets/KALE_datasets/fb122/rel_idx.p', 'rb'))
id2rel_dict = rel_idx['id2rel_dict']; rel2id_dict = rel_idx['rel2id_dict']; 


#2. Check intersect exists with rank(P1,P2_vertical)=rank(r_1, r_2) < # rows between all pairs of r's
from itertools import combinations 
comb = combinations(range(122),2)
intersect_yes_or_no = {}
for r1, r2 in list(comb):
    #vv_r1 = proj_all_r[r1]; vv_r2 = proj_all_r[r2]
    vv_r1r2 = torch.cat([proj_all_r[r1],proj_all_r[r2]], dim=0).view(-1, 1)
    P1P2 = torch.cat([unique_projmat[r1], unique_projmat[r2]], dim = 0) #shape is [2*embd_dim x embed_dim]
    P1P2_augmented = torch.cat([P1P2, vv_r1r2], dim=1)
    P1P2_rank = np.linalg.matrix_rank(P1P2.cpu().data.numpy())
    P1P2_augmented_rank = np.linalg.matrix_rank(P1P2_augmented.cpu().data.numpy())
    if P1P2_rank == P1P2_augmented_rank:
        intersect_yes_or_no[(r1, r2)] =  1
    else:
        intersect_yes_or_no[(r1, r2)] =  0

ints = [k for k,v in intersect_yes_or_no.items() if v ==1]  
ints_in_eng = [(id2rel_dict[pair[0]], id2rel_dict[pair[1]])for pair in ints]
three_bundles = [id2rel_dict[x] for x in [3, 26,34,49,45]]
zero_bundles = [id2rel_dict[x] for x in [0,82,48,51,42,58]]

#3. Find null space of the P's and do +r's (or -r's)
from scipy.linalg import null_space, subspace_angles
null_spaces = {}
for r in range(122):
    null = null_space(unique_projmat[r].data.cpu().numpy())
    if null.shape[1] != 0:
        null_spaces[r] = null + np.tile(rel_embeddings[r].reshape((-1,1)), null.shape[1] ) #each column is in null space of Px = Pr
    else:
        null_spaces[r] =  rel_embeddings[r].reshape((-1,1))

#4. Find angle 
comb = combinations(range(122),2)
angles = {}
for r1, r2 in list(comb):
    angles[(r1,r2)] = np.rad2deg(subspace_angles(null_spaces[r1], null_spaces[r2]))[0].item()
place_of_birth = rel2id_dict['/people/person/place_of_birth']
nationality = rel2id_dict['/people/person/nationality']
rel_angles = {(id2rel_dict[k[0]], id2rel_dict[k[1]]):v for k ,v in angles.items()}

sorted_angles = sorted(angles.items(), key=lambda kv: kv[1])
sorted_rel_angles = sorted(rel_angles.items(), key=lambda kv: kv[1])
import collections
sorted_angles = collections.OrderedDict(sorted_angles)
sorted_rel_angles = collections.OrderedDict(sorted_rel_angles)

pickle.dump(sorted_rel_angles, open('/data/scratch-oc40/symin95/thesis_kg_project/transint/knowledge_representation_pytorch-master/model/KALE/FB122/dim_100_hr_gap1_no_norm_ent_f0imp1/l_0.005_es_1000_L_1_em_100_nb_100_n_1000_m_5.0_f_0_mo_0.9_s_0_op_1_lo_0/sorted_rel_angles_for.p', 'wb'))

#5. Find projection : Do it on local computer! 

perc_30 = len([sorted_rel_angles[k] for k in sorted_rel_angles if sorted_rel_angles[k]<30]) / len(sorted_rel_angles)

#Do the distance equations 

#Identify all (h,t,r)'s for  

place_of_birth = rel2id_dict['/people/person/place_of_birth']

lived_in = rel2id_dict['/people/person/places_lived./people/place_lived/location']
nationality = rel2id_dict['/people/person/nationality']
cause_of_death = rel2id_dict['/people/cause_of_death/people']
sports_colors = rel2id_dict['/sports/sports_team/colors']

#load test data and find all the triples that belong to the relations
pos_triples = pickle.load(open('datasets/KALE_datasets/fb122/test_triples_list.p', 'rb'))

triple_idx2_triple = {}
i=0
for pos_triple in pos_triples:
    triple_idx2_triple[i] = pos_triple
    i+=1

pos_rel_idx_dict = {}
for i, rel in enumerate([pair[2] for pair in pos_triples]):
    if not(rel in pos_rel_idx_dict):
        pos_rel_idx_dict[rel] = []
    pos_rel_idx_dict[rel].append(i)


def triple_classification(mod, pos_h, pos_t, orig_r, proj_r):
    pos_h_e = mod.ent_embeddings(pos_h)
    pos_t_e = mod.ent_embeddings(pos_t)
    #pos_r_e = self.rel_embeddings(pos_r) #\vv{r}
    pos_sub = pos_h_e - pos_t_e
    num_ent = pos_h_e.shape[0]
    #unique_time = time.time()
    #print("unique time:", unique_time-time.time())
    #comb  = time.time()
    #This has shape [#unique_rel x 2 x emb_dim]
    unique_bases = torch.tensor(null_spaces[proj_r]).cuda().t()#this is AT, A is unique_bases.t()
    orig_bases = torch.tensor(null_spaces[orig_r]).cuda().t()
    #this will have shape [#unique_rel x 2]
    ####THIS HERE IS WRONG
            #print("comb time :", time.time()-comb)
    #inv_time = time.time()
    ATA_inverse = torch.inverse(torch.mm(unique_bases, unique_bases.t())).cuda()
    orig_ATA_inverse = torch.inverse(torch.mm(orig_bases, orig_bases.t())).cuda()
    #print("ATA inverse time :", time.time()-inv_time)
    #proj_time = time.time()
    unique_projmat = to_cuda(torch.eye(mod.embedding_size).type(floatTensor)) - torch.mm(torch.mm(unique_bases.t(), ATA_inverse), unique_bases).type(floatTensor)
    orig_unique_projmat = to_cuda(torch.eye(mod.embedding_size).type(floatTensor)) - torch.mm(torch.mm(orig_bases.t(), orig_ATA_inverse), orig_bases).type(floatTensor)
    orig_P_pos_r =  torch.cat([orig_unique_projmat.unsqueeze(0)]*pos_h_e.shape[0], dim=0)
    P_pos_r = torch.cat([unique_projmat.unsqueeze(0)]*pos_h_e.shape[0], dim=0)
    #need to fix here
    pos_sub =  torch.bmm(orig_P_pos_r, pos_sub.view(num_ent,mod.embedding_size,1)).squeeze(2) 
    bmmed = torch.bmm(P_pos_r, pos_sub.view(num_ent,mod.embedding_size,1)).squeeze(2) 
    #print("Time for bmm-ing:", time.time()-bmm_time)
    #print("time for bmm:", time.time() - bmm_tim)
    proj_sub = bmmed[:num_ent]
    dist_l1 = torch.sum(torch.abs(proj_sub- pos_sub), 1)
    dist_l2 = torch.sum((proj_sub - pos_sub) ** 2, 1)
    return dist_l1, dist_l2

birth_h = torch.tensor([triple_idx2_triple[trip][0] for trip in pos_rel_idx_dict[place_of_birth]]).cuda()
birth_t = torch.tensor([triple_idx2_triple[trip][1] for trip in pos_rel_idx_dict[place_of_birth]]).cuda()

lived_in_h = torch.tensor([triple_idx2_triple[trip][0] for trip in pos_rel_idx_dict[lived_in]]).cuda()
lived_in_t = torch.tensor([triple_idx2_triple[trip][1] for trip in pos_rel_idx_dict[lived_in]]).cuda()

orig_r = lived_in; proj_r = place_of_birth
lived_in_proj_to_birth_place_dist1, lived_in_proj_to_birth_place_dist2 = triple_classification(mod, lived_in_h, lived_in_t, orig_r, proj_r)  

orig_r = place_of_birth; proj_r = lived_in
birth_place_proj_to_live_in_dist1, birth_place_proj_to_live_in_dist2 = triple_classification(mod, birth_h, birth_t,orig_r, proj_r)  

orig_r = place_of_birth; proj_r = nationality
birth_place_proj_to_nationality_dist1, birth_place_proj_to_nationality_dist2 = triple_classification(mod, birth_h, birth_t, orig_r, proj_r)  

nat_t = torch.tensor([triple_idx2_triple[trip][0] for trip in pos_rel_idx_dict[nationality]]).cuda()
nat_h =  torch.tensor([triple_idx2_triple[trip][1] for trip in pos_rel_idx_dict[nationality]]).cuda()

orig_r = nationality; proj_r = place_of_birth
nationality_to_birthplace_dist1, nationality_to_birthplace_dist2 = triple_classification(mod, nat_h, nat_t, orig_r, proj_r)  

a = torch.mean(lived_in_proj_to_birth_place_dist2)
b = torch.mean(birth_place_proj_to_live_in_dist2)

imp = 0.5 * (a/b + b/a)

a = torch.mean(lived_in_proj_to_birth_place_dist1)
b = torch.mean(birth_place_proj_to_live_in_dist1)

imp = 0.5 * (a/b + b/a)

a = torch.mean(nationality_to_birthplace_dist2)
b = torch.mean(birth_place_proj_to_nationality_dist2)

no_imp = 0.5 * (a/b + b/a)