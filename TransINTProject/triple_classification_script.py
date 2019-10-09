#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:07:19 2019

@author: TiffMin
"""

#########
##### Triple Classification Script
########

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


def triple_classification(mod, pos_h, pos_t, pos_r):
    pos_h_e = mod.ent_embeddings(pos_h)
    pos_t_e = mod.ent_embeddings(pos_t)
    #pos_r_e = self.rel_embeddings(pos_r) #\vv{r}
    pos_r_e = mod.vvrel_embedding_func(pos_r)
    pos_sub = pos_h_e + pos_r_e - pos_t_e
    
    num_ent = pos_h_e.shape[0]
    #unique_time = time.time()
    unique_rels_2_unique_proj_idx = mod.unique_rels(pos_r)
    #print("unique time:", unique_time-time.time())
    unique_rel_tensor = to_cuda(longTensor([int(key) for key in unique_rels_2_unique_proj_idx]))
    #comb  = time.time()
    #This has shape [#unique_rel x 2 x emb_dim]
    unique_bases = mod.comb_mats(unique_rel_tensor) #this is AT, A is unique_bases.t()
    #this will have shape [#unique_rel x 2]
    ####THIS HERE IS WRONG
            #print("comb time :", time.time()-comb)
    #inv_time = time.time()
    ATA_inverse = mod.inverse(torch.bmm(unique_bases, unique_bases.transpose(1,2)))
    #print("ATA inverse time :", time.time()-inv_time)
    #proj_time = time.time()
    unique_projmat = to_cuda(torch.cat([torch.eye(mod.embedding_size)]*len(unique_rels_2_unique_proj_idx)).view(len(unique_rels_2_unique_proj_idx), mod.embedding_size, mod.embedding_size)) - torch.bmm(torch.bmm(unique_bases.transpose(1,2), ATA_inverse), unique_bases)
    #print("Multiplying for projection :", time.time()-proj_time)
    #Projection Matrices 
    #Make in the shape of num_ent x emb_dim x emb_dim  (every proj matrix at every num_ent)
    #assign_time = time.time()
    unique_proj_idx = [unique_rels_2_unique_proj_idx[pos_r[i].item()] for i in range(num_ent)] 
    P_pos_r = unique_projmat[unique_proj_idx]
    #need to fix here
    bmmed = torch.bmm(P_pos_r, pos_sub.view(num_ent,mod.embedding_size,1)).squeeze(2) 
    #print("Time for bmm-ing:", time.time()-bmm_time)
    #print("time for bmm:", time.time() - bmm_tim)
    pos_sub = bmmed[:num_ent]
    if mod.L1_flag:
        pos = torch.sum(torch.abs(pos_sub), 1)
    else:
        pos = torch.sum((pos_sub) ** 2, 1)
    return pos

#do it by relation
def pos_acc_by_rel(pos_scores, pos_rel_idx_dict, thr):
    score_by_rel = {rel:None for rel in pos_rel_idx_dict}
    len_by_rel = {rel:len(pos_rel_idx_dict[rel]) for rel in pos_rel_idx_dict}
    for rel in pos_rel_idx_dict:
        score_by_rel[rel] = sum((pos_scores[pos_rel_idx_dict[rel]] < thr).type(floatTensor)) 
    return score_by_rel, len_by_rel
    
def neg_acc_by_rel(neg_scores, neg_rel_idx_dict, thr):
    score_by_rel = {rel:None for rel in neg_rel_idx_dict}
    len_by_rel = {rel:len(neg_rel_idx_dict[rel]) for rel in neg_rel_idx_dict}
    for rel in neg_rel_idx_dict:
        score_by_rel[rel] = sum((neg_scores[neg_rel_idx_dict[rel]] >= thr).type(floatTensor)) 
    return score_by_rel, len_by_rel
    



PATH = '/data/scratch-oc40/symin95/thesis_kg_project/transint/knowledge_representation_pytorch-master/model/KALE/FB122/dim_100_hr_gap1_no_norm_ent_f1imp1/l_0.005_es_1000_L_1_em_100_nb_100_n_1000_m_10.0_f_1_mo_0.9_s_0_op_1_lo_0/TransINT.ckpt'
PATH_imp = '/data/scratch-oc40/symin95/thesis_kg_project/transint/knowledge_representation_pytorch-master/model/KALE/FB122/dim_100_hr_gap1_no_norm_ent_f1imp1/l_0.003_es_1000_L_1_em_100_nb_100_n_1000_m_5.0_f_1_mo_0.9_s_0_op_1_lo_0/TransINT.ckpt'
PATH_no_imp = '/data/scratch-oc40/symin95/thesis_kg_project/transint/knowledge_representation_pytorch-master/model/KALE/FB122/dim_100_hr_gap1_no_norm_ent_f1imp0/l_0.005_es_1000_L_1_em_100_nb_100_n_1000_m_10.0_f_1_mo_0.9_s_0_op_1_lo_0/TransINT_epoch_870.ckpt'
mod=torch.load(PATH_no_imp)
mod.eval()
mod = mod.cuda() 

#1: Pull out parameters for the P's and r's 
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

#2. Load Pos and Neg triples 
pos_triples = pickle.load(open('datasets/KALE_datasets/fb122/test_triples_list.p', 'rb'))
neg_triples = pickle.load(open('datasets/KALE_datasets/fb122/neg_test_triples_list.p', 'rb'))

all_triples = {}
triple_idx2_triple = {}
i=0
for pos_triple in pos_triples:
    all_triples[pos_triple] = +1
    triple_idx2_triple[i] = pos_triple
    i+=1
for neg_triple in neg_triples:
    all_triples[neg_triple] = -1
    triple_idx2_triple[i] = neg_triple
    i+=1

scores_for_triple_idxes = {}
posheadList = torch.tensor([pair[0] for pair in pos_triples]).cuda()
postailList = torch.tensor([pair[1] for pair in pos_triples]).cuda()
posrelList = torch.tensor([pair[2] for pair in pos_triples]).cuda()

negheadList = torch.tensor([pair[0] for pair in neg_triples]).cuda()
negtailList = torch.tensor([pair[1] for pair in neg_triples]).cuda()
negrelList = torch.tensor([pair[2] for pair in neg_triples]).cuda()

pos_scores = triple_classification(mod, posheadList, postailList, posrelList)
neg_scores = triple_classification(mod, negheadList, negtailList, negrelList)

import itertools
thresholds = [[i, i+0.25, i+0.5, i+0.75] for i in range(20, 80)]
thresholds = list(itertools.chain.from_iterable(thresholds))

total = len(posheadList) + len(negheadList)
threshold_score = {}
cur_best = 0
cur_best_thr = None

pos_rel_idx_dict = {}
neg_rel_idx_dict = {}
for i, rel in enumerate([pair[2] for pair in pos_triples]):
    if not(rel in pos_rel_idx_dict):
        pos_rel_idx_dict[rel] = []
    pos_rel_idx_dict[rel].append(i)
for i, rel in enumerate([pair[2] for pair in neg_triples]):
    if not(rel in neg_rel_idx_dict):
        neg_rel_idx_dict[rel] = []
    neg_rel_idx_dict[rel].append(i)


for thr in thresholds:
    pos_score_by_rel, pos_len_by_rel = pos_acc_by_rel(pos_scores, pos_rel_idx_dict, thr)
    neg_score_by_rel, neg_len_by_rel =  neg_acc_by_rel(neg_scores, neg_rel_idx_dict, thr)
    threshold_score[thr] = torch.mean(torch.tensor([(pos_score_by_rel[rel] + neg_score_by_rel[rel]/10) / (2*pos_len_by_rel[rel]) for rel in pos_score_by_rel]))
    if threshold_score[thr]> cur_best:
        cur_best = threshold_score[thr]
        cur_best_thr = thr

thr = cur_best_thr
pos_score_by_rel, pos_len_by_rel = pos_acc_by_rel(pos_scores, pos_rel_idx_dict, thr)
neg_score_by_rel, neg_len_by_rel =  neg_acc_by_rel(neg_scores, neg_rel_idx_dict, thr)
acc_by_dict = {}
for i, item in enumerate([(pos_score_by_rel[rel] + neg_score_by_rel[rel]/10) / (2*pos_len_by_rel[rel]) for rel in pos_score_by_rel]):
    acc_by_dict[i] = item.cpu().data.item()
pickle.dump(acc_by_dict, open('datasets/KALE_datasets/fb122/acc_by_dict.p', 'wb'))

rank_start = pickle.load(open('/data/scratch-oc40/symin95/thesis_kg_project/transint/knowledge_representation_pytorch-master/datasets/KALE_datasets/fb122/NULL_dim_20_hr_gap1_dict.p', 'rb'))['rank_start']
in_bundle = copy.deepcopy(list(rank_start.keys()))

in_bundle_acc = np.mean([acc_by_dict[i] for i in acc_by_dict if i in in_bundle])
not_in_bundle_acc = np.mean([acc_by_dict[i] for i in acc_by_dict if not( i in in_bundle)])

