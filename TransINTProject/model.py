#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-04 11:37:04
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

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

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor

class TransEModel(nn.Module):
    def __init__(self, config):
        super(TransEModel, self).__init__()
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size

        ent_weight = floatTensor(self.entity_total, self.embedding_size)
        rel_weight = floatTensor(self.relation_total, self.embedding_size) #num_rel x embedding size
        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform(ent_weight) #don't do xavier but the ones in the TransE paper (root 6 whatever uniform and then normalize) 
        nn.init.xavier_uniform(rel_weight) 
        rel_weight = rel_weight.div(torch.norm(rel_weight,dim=1).unsqueeze(1).expand_as(rel_weight)) #divide with the norm of each embedding vector
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h) #batch_size x embedding size
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        
        #normalize entity embeddings only
        pos_h_e = pos_h_e.div(torch.norm(pos_h_e,dim=1).unsqueeze(1).expand_as(pos_h_e)) #divide with the norm of each embedding vector
        pos_t_e = pos_t_e.div(torch.norm(pos_t_e,dim=1).unsqueeze(1).expand_as(pos_t_e)) 
        neg_h_e = neg_h_e.div(torch.norm(neg_h_e,dim=1).unsqueeze(1).expand_as(neg_h_e)) 
        neg_t_e = neg_t_e.div(torch.norm(neg_t_e,dim=1).unsqueeze(1).expand_as(neg_t_e)) 

        # L1 distance
        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1) #torch.abs까지는 batch_size x embedding size #torch.sum에서 dim=1 이기
                                                                        #때문에 summed over elements of embedding vectors, ouput is [batch_size]
                                                                        #이고 각각의 element is sum of element of each embedding vector 
                                                                        #which is l1 distance 
                                                                        #output is [distance * batch_size]
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1) 
        # L2 distance
        else:
            #pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            #neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
            pos = torch.sqrt(torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1))
            neg = torch.sqrt(torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1))
        return pos, neg

class TransHModel(nn.Module):
    def __init__(self, config):
        super(TransHModel, self).__init__()
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size

        ent_weight = floatTensor(self.entity_total, self.embedding_size)
        rel_weight = floatTensor(self.relation_total, self.embedding_size)
        norm_weight = floatTensor(self.relation_total, self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        nn.init.xavier_uniform(norm_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.norm_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.norm_embeddings.weight = nn.Parameter(norm_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.norm_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb
        self.norm_embeddings.weight.data = normalize_norm_emb

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_norm = self.norm_embeddings(pos_r)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        neg_norm = self.norm_embeddings(neg_r)

        pos_h_e = projection_transH_pytorch(pos_h_e, pos_norm)
        pos_t_e = projection_transH_pytorch(pos_t_e, pos_norm)
        neg_h_e = projection_transH_pytorch(neg_h_e, neg_norm)
        neg_t_e = projection_transH_pytorch(neg_t_e, neg_norm)

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg

# TransR without using pretrained embeddings,
# i.e, the whole model is trained from scratch.
class TransRModel(nn.Module):
    def __init__(self, config):
        super(TransRModel, self).__init__()
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.ent_embedding_size = config.ent_embedding_size
        self.rel_embedding_size = config.rel_embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size

        ent_weight = floatTensor(self.entity_total, self.ent_embedding_size)
        rel_weight = floatTensor(self.relation_total, self.rel_embedding_size)
        proj_weight = floatTensor(self.relation_total, self.rel_embedding_size * self.ent_embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        nn.init.xavier_uniform(proj_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.ent_embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size)
        self.proj_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size * self.ent_embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.proj_embeddings.weight = nn.Parameter(proj_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        # normalize_proj_emb = F.normalize(self.proj_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb
        # self.proj_embeddings.weight.data = normalize_proj_emb

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_proj = self.proj_embeddings(pos_r)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        neg_proj = self.proj_embeddings(neg_r)

        pos_h_e = projection_transR_pytorch(pos_h_e, pos_proj)
        pos_t_e = projection_transR_pytorch(pos_t_e, pos_proj)
        neg_h_e = projection_transR_pytorch(neg_h_e, neg_proj)
        neg_t_e = projection_transR_pytorch(neg_t_e, neg_proj)

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e    

# TransR with using pretrained embeddings.
# Pretrained embeddings are trained with TransE, stored in './transE_%s_%s_best.pkl',
# with first '%s' dataset name,
# second '%s' embedding size.
# Initialize projection matrix with identity matrix.
class TransRPretrainModel(nn.Module):
    def __init__(self, config):
        super(TransRPretrainModel, self).__init__()
        self.dataset = config.dataset
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.ent_embedding_size = config.ent_embedding_size
        self.rel_embedding_size = config.rel_embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size

        with open('./transE_%s_%s_best.pkl' % (config.dataset, str(config.ent_embedding_size)), 'rb') as fr:
            ent_embeddings_list = pickle.load(fr)
            rel_embeddings_list = pickle.load(fr)

        ent_weight = floatTensor(ent_embeddings_list)
        rel_weight = floatTensor(rel_embeddings_list)
        proj_weight = floatTensor(self.rel_embedding_size, self.ent_embedding_size)
        nn.init.eye(proj_weight)
        proj_weight = proj_weight.view(-1).expand(self.relation_total, -1)

        self.ent_embeddings = nn.Embedding(self.entity_total, self.ent_embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size)
        self.proj_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size * self.ent_embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.proj_embeddings.weight = nn.Parameter(proj_weight)

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_proj = self.proj_embeddings(pos_r)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        neg_proj = self.proj_embeddings(neg_r)

        pos_h_e = projection_transR_pytorch(pos_h_e, pos_proj)
        pos_t_e = projection_transR_pytorch(pos_t_e, pos_proj)
        neg_h_e = projection_transR_pytorch(neg_h_e, neg_proj)
        neg_t_e = projection_transR_pytorch(neg_t_e, neg_proj)

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e

# TransD with using pretrained embeddings, 
# and embeddings of entities and relations are of the same size.
# It can be viewed as a special case of TransH,
# (See "Knowledge Graph Embedding via Dynamic Mapping Matrix" paper)
# Pretrained embeddings are trained with TransE, stored in './transE_%s_%s_best.pkl',
# with first '%s' dataset name,
# second '%s' embedding size.
# Initialize projection matrices with zero matrices.
class TransDPretrainModelSameSize(nn.Module):
    def __init__(self, config):
        super(TransDPretrainModelSameSize, self).__init__()
        self.dataset = config.dataset
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size

        with open('./transE_%s_%s_best.pkl' % (config.dataset, str(config.embedding_size)), 'rb') as fr:
            ent_embeddings_list = pickle.load(fr)
            rel_embeddings_list = pickle.load(fr)

        ent_weight = floatTensor(ent_embeddings_list)
        rel_weight = floatTensor(rel_embeddings_list)
        ent_proj_weight = floatTensor(self.entity_total, self.embedding_size)
        rel_proj_weight = floatTensor(self.relation_total, self.embedding_size)
        ent_proj_weight.zero_()
        rel_proj_weight.zero_()

        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_proj_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_proj_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.ent_proj_embeddings.weight = nn.Parameter(ent_proj_weight)
        self.rel_proj_embeddings.weight = nn.Parameter(rel_proj_weight)

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_h_proj = self.ent_proj_embeddings(pos_h)
        pos_t_proj = self.ent_proj_embeddings(pos_t)
        pos_r_proj = self.rel_proj_embeddings(pos_r)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        neg_h_proj = self.ent_proj_embeddings(neg_h)
        neg_t_proj = self.ent_proj_embeddings(neg_t)
        neg_r_proj = self.rel_proj_embeddings(neg_r)

        pos_h_e = projection_transD_pytorch_samesize(pos_h_e, pos_h_proj, pos_r_proj)
        pos_t_e = projection_transD_pytorch_samesize(pos_t_e, pos_t_proj, pos_r_proj)
        neg_h_e = projection_transD_pytorch_samesize(neg_h_e, neg_h_proj, neg_r_proj)
        neg_t_e = projection_transD_pytorch_samesize(neg_t_e, neg_t_proj, neg_r_proj)

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e

class TransINTModel(nn.Module):
    def __init__(self, config, rank_file, r_id2_vrr, equiv_flip_dict):
        super(TransINTModel, self).__init__()
        #there are really just two parameters:
        #self.vvr_bundle_heads, self.dict_of_random_mats 
        #Parameter
        self.rank_start = rank_file['rank_start']
        self.rank_end = rank_file['rank_end']
        #self.vvr_bundle_heads = nn.ParameterDict({str(k): self.param_cuda(v) for k, v in rank_file['vvr_bundle_heads'].items()} )
        self.vvr_bundle_heads_embs =  nn.Parameter(torch.cat([to_cuda(v) for k,v in rank_file['vvr_bundle_heads'].items()]))
        self.vvr_bundle_heads_keys = {}
        i=0
        for k in rank_file['vvr_bundle_heads']:
            self.vvr_bundle_heads_keys[k] = i
            i+=1
        del i
        self.dict_of_linenar_comb = nn.ParameterDict({str(k): self.param_cuda(v) for k, v in rank_file['dict_of_linenar_comb'].items()} )
        
        #offset lin. comb's only need to be there for the bunddle heads
        #so same number as vvr_bundle_heads_embs's keys 
        self.offset_lin_combs = self.param_cuda(nn.Parameter(torch.randn(len(rank_file['vvr_bundle_heads']), 2)))
        self.offset_keys =  self.vvr_bundle_heads_keys #same as vvr bundle heads keys
        
        #self.offset_embs =  nn.Parameter(torch.cat([to_cuda(v) for k,v in rank_file['dict_of_offset'].items()]))
        #self.offset_keys =  self.vvr_bundle_heads_keys #same as vvr bundle heads keys
        #self.dict_of_offset = nn.ParameterDict({str(k): self.param_cuda(v) for k, v in rank_file['dict_of_offset'].items()} )
        
        #Parameter
        self.dict_of_random_mats = nn.ParameterDict({str(k):self.param_cuda(v) for k, v in rank_file['dict_of_random_mats'].items()})
        #self.dict_of_masks = {k: (to_cuda(v[0]), to_cuda(v[1])) for k, v in rank_file['dict_of_masks'].items()}
        self.r_id2_vrr = r_id2_vrr
        self.equiv_flip_dict = equiv_flip_dict
        self.equiv_flip_dict_both_sides = copy.deepcopy(equiv_flip_dict)
        for k, v in self.equiv_flip_dict.items():
            self.equiv_flip_dict_both_sides[v]=k
        
        self.which_dataset = config.which_dataset
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size
        
        #entity embeddings
        vv_ent_weight = floatTensor(self.entity_total, self.embedding_size)
        nn.init.xavier_uniform_(vv_ent_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(vv_ent_weight)
        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_entity_emb =  normalize_entity_emb
        self.ent_embeddings.weight.data = normalize_entity_emb
        
    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        #pos_r_e = self.rel_embeddings(pos_r) #\vv{r}
        pos_r_e = self.vvrel_embedding_func(pos_r)
        pos_sub = pos_h_e + pos_r_e - pos_t_e

        neg_h_e = self.ent_embeddings(neg_h)
        #embed_time= time.time()
        neg_t_e = self.ent_embeddings(neg_t)
        #print("embed time :", time.time()-embed_time)
        #vvrel_time= time.time()
        #print("vvrel time :", time.time()-vvrel_time)
        neg_sub = neg_h_e + pos_r_e - neg_t_e #neg_r_e == pos_r_e
        
        num_ent = pos_h_e.shape[0]
        #unique_time = time.time()
        unique_rels_2_unique_proj_idx = self.unique_rels(pos_r)
        #print("unique time:", unique_time-time.time())
        unique_rel_tensor = to_cuda(longTensor([int(key) for key in unique_rels_2_unique_proj_idx]))
        #comb  = time.time()
        #This has shape [#unique_rel x 2 x emb_dim]
        unique_bases = self.comb_mats(unique_rel_tensor) #this is AT, A is unique_bases.t()
        #this will have shape [#unique_rel x 2]
        ####THIS HERE IS WRONG
        offset_access_index = [self.vvr_bundle_heads_keys[self.r_id2_vrr[rel.item()][0]] for rel in unique_rel_tensor]
        mult = torch.tensor([self.r_id2_vrr[rel.item()][1] for rel in unique_rel_tensor]).type(torch.FloatTensor).cuda() #shape: [unique_rel]
        #These are only for unique rels
        offset_lin_combs_for_unique_rels = self.offset_lin_combs[offset_access_index] 
        #This will have shape [#unique_rel x emb_dim]
        #unique_offset = torch.bmm(offset_lin_combs_for_unique_rels.unsqueeze(1), unique_bases).squeeze(1)
        #unique_offset = unique_offset * torch.cat([mult.view(-1,1)] * self.embedding_size, dim=1)
        #print("comb time :", time.time()-comb)
        #inv_time = time.time()
        ATA_inverse = self.inverse(torch.bmm(unique_bases, unique_bases.transpose(1,2)))
        #print("ATA inverse time :", time.time()-inv_time)
        #proj_time = time.time()
        unique_projmat = to_cuda(torch.cat([torch.eye(self.embedding_size)]*len(unique_rels_2_unique_proj_idx)).view(len(unique_rels_2_unique_proj_idx), self.embedding_size, self.embedding_size)) - torch.bmm(torch.bmm(unique_bases.transpose(1,2), ATA_inverse), unique_bases)
        #print("Multiplying for projection :", time.time()-proj_time)
        #Projection Matrices 
        #Make in the shape of num_ent x emb_dim x emb_dim  (every proj matrix at every num_ent)
        #assign_time = time.time()
        
        unique_proj_idx = [unique_rels_2_unique_proj_idx[pos_r[i].item()] for i in range(num_ent)] *2
        P_pos_r = unique_projmat[unique_proj_idx]
        #need to fix here
        #r_offsets = unique_offset[[unique_rels_2_unique_proj_idx[pos_r[i].item()] for i in range(num_ent)]]
        
# =============================================================================
#         P_pos_r = to_cuda(torch.zeros(num_ent*2, self.embedding_size, self.embedding_size))
#         for i in range(num_ent):
#             rel = pos_r[i].item()
#             unique_proj_idx = unique_rels_2_unique_proj_idx[rel]
#             P_pos_r[i] = unique_projmat[unique_proj_idx]; P_pos_r[i+num_ent] = unique_projmat[unique_proj_idx]
# =============================================================================
        #print("Time for assining projection to P_Pos_r:", time.time()-assign_time)
        #Apply projection to the entity vector embeddngs
        #Make column vectors in the shape of num_ent x emb_dim x 1
        
        #bmm_tim = time.time()
        #Going to have shape num_ent x emb_dim x 1 when came out of the bmm
        #bmm_time = time.time()
        bmmed = torch.bmm(P_pos_r, torch.cat([pos_sub, neg_sub]).view(2*num_ent,self.embedding_size,1)).squeeze(2) 
        #print("Time for bmm-ing:", time.time()-bmm_time)

        #print("time for bmm:", time.time() - bmm_tim)
        sum_time = time.time()
        pos_sub = bmmed[:num_ent]
        neg_sub = bmmed[num_ent:2*num_ent]

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_sub), 1)
            neg = torch.sum(torch.abs(neg_sub), 1)
        else:
            pos = torch.sum((pos_sub) ** 2, 1)
            neg = torch.sum((neg_sub) ** 2, 1)
        #print("time for sum", time.time() - sum_time)
        return pos, neg
    
    #
    def comb_mats(self, rel_tensor, in_batch=True, cpu=False):
        if self.which_dataset in ['wn18', 'fb122', 'Nell_loc']:
            stack_mats = to_cuda(torch.zeros(rel_tensor.shape[-1], 2, self.embedding_size))
        elif self.which_dataset in ['Nell_sp']:
            stack_mats = to_cuda(torch.zeros(rel_tensor.shape[-1], 3, self.embedding_size))
        
        i=0
        for rel_tent in rel_tensor:
            rel = rel_tent.item()
            if rel in self.equiv_flip_dict:
                rel = self.equiv_flip_dict[rel]
            p_rel = self.r_id2_vrr[rel][0]
            #no need to stack 
            if p_rel == rel or (self.rank_start[rel] ==self.rank_start[p_rel] and self.rank_end[rel] == self.rank_end[p_rel]):
                if self.rank_end[rel] ==0:
                    result = to_cuda(torch.zeros(2, self.embedding_size))
                    result[0] = self.dict_of_random_mats[str(p_rel)]
                else:
                    result = self.dict_of_random_mats[str(p_rel)]
                if cpu==True:
                    stack_mats[i] = result.cpu()
                stack_mats[i] = result 
            else: 
                #stack everything from above and return 
                result = torch.mm(self.dict_of_linenar_comb[str(rel)], self.dict_of_random_mats[str(p_rel)])
                result_num_rank = result.shape[0]
                if  cpu==True:
                    stack_mats[i][:result_num_rank] = result.cpu()
                stack_mats[i][:result_num_rank] = result
            i+=1
        return stack_mats

    def inverse(self, tensor_3d):
        inverse_tensor = to_cuda(torch.zeros(tensor_3d.shape[0], tensor_3d.shape[1], tensor_3d.shape[2]))
        for i in range(tensor_3d.shape[0]):
            #assert (tensor_3d[i].shape[0] == 2 and tensor_3d[i].shape[1] == 2)
            if self.which_dataset in ['wn18', 'fb122', 'Nell_loc']:
                a = tensor_3d[i,0,0]; b = tensor_3d[i,0,1]; c = tensor_3d[i,1,0]; d = tensor_3d[i,1,1]
                det = a*d-b*c
                if det !=0:
                    #inverse_tensor[i] = torch.inverse(tensor_3d[i])
                    #try a new inverse   
                    inverse_tensor[i] = to_cuda(torch.tensor([[d, -b], [-c, a]]))*1/(det)
                else:
                    #print("inverse", 1/tensor_3d[i][0][0])
                    inverse_tensor[i] = to_cuda(torch.eye(tensor_3d.shape[1])) * 1/a
            elif self.which_dataset in ['Nell_sp']:
                m1, m2, m3, m4, m5, m6, m7, m8, m9 = tensor_3d[i,0,0], tensor_3d[i,0,1], tensor_3d[i,0,2], tensor_3d[i,1,0], tensor_3d[i,1,1], tensor_3d[i,1,2], tensor_3d[i,2,0],tensor_3d[i,2,1], tensor_3d[i,2,2]
                det = m1*m5*m9 + m4*m8*m3 + m7*m2*m6 - m1*m6*m8 - m3*m5*m7 - m2*m4*m9  
                if det !=0:
                    inverse_tensor[i] = to_cuda( torch.tensor([[m5*m9-m6*m8, m3*m8-m2*m9, m2*m6-m3*m5],
                     [m6*m7-m4*m9, m1*m9-m3*m7, m3*m4-m1*m6],
                     [m4*m8-m5*m7, m2*m7-m1*m8, m1*m5-m2*m4]]))/det
                else:
                    det = m1*m5 - m2*m4
                    if det !=0:   
                        inverse_tensor[i] = to_cuda(torch.tensor([[m5, -m2,0.0], [-m4, m1,0.0], [0.0, 0.0, 0.0]]))*1/(det)
                    else:
                        inverse_tensor[i] = to_cuda(torch.eye(tensor_3d.shape[1])) * 1/m1
        return inverse_tensor
    
    
    def vvrel_embedding_func(self, rel_tensors):
        access_indexes = [self.vvr_bundle_heads_keys[self.r_id2_vrr[rel.item()][0]] for rel in rel_tensors]
        mult = torch.tensor([self.r_id2_vrr[rel.item()][1] for rel in rel_tensors]).type(torch.FloatTensor).cuda()
        #print("access_indexes", access_indexes)
        #print("vvr bundleheads", self.vvr_bundle_heads_embs[access_indexes].shape)
        #print("mult", mult.shape)
        return_vvrs = self.vvr_bundle_heads_embs[access_indexes] * torch.cat([mult.view(-1,1)] *self.embedding_size, dim=1)
        #return_offsets = self.offset_embs[access_indexes] * torch.cat([mult.view(-1,1)] *50, dim=1)
# =============================================================================
#         return_tensor = to_cuda(torch.zeros(rel_tensors.shape[0], self.embedding_size)) #cuda?
#         i=0
#         for rel_tent in rel_tensors:
#             rel = rel_tent.item()
#             return_tensor[i] = self.vvr_bundle_heads[str(self.r_id2_vrr[rel][0])] * self.r_id2_vrr[rel][1]
#             i+=1
# =============================================================================
        return return_vvrs#, return_offsets
                
    def vvrel_all_weights(self):
        return_np = np.zeros((self.relation_total, self.embedding_size))
        for rel in range(self.relation_total):
            #return_np[i] = v.data.cpu().numpy()
            return_np[rel]= self.vvrel_embedding_func(to_cuda(torch.tensor([rel]))).data.cpu().numpy()
        return return_np
    
    def param_cuda(self, param):
        return nn.Parameter(to_cuda(param), requires_grad = True)
    
    def unique_rels(self, rel_tensors):
        unique_rels_2_unique_proj_idx = {}
        i = 0
        for rel_tent in rel_tensors:
            rel = rel_tent.item()
            if not(rel in unique_rels_2_unique_proj_idx):
                unique_rels_2_unique_proj_idx[rel] =i
                i+=1
        return unique_rels_2_unique_proj_idx
    
    def normalize_vvrel_embeddings(self):
        normalized = F.normalize(self.vvr_bundle_heads_embs.data, p=2, dim=1) #101x50
        self.vvr_bundle_heads_embs.data = normalized
        normalized = F.normalize(self.offset_lin_combs.data, p=2, dim=1)
        self.offset_lin_combs.data = normalized
        
    def normalize_ent_embeddings(self):
        normalized = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1) #101x50
        self.ent_embeddings.weight.data = normalized
        
    def normalize_all_rows(self):
        for k in self.dict_of_random_mats:
            self.dict_of_random_mats[k] = self.param_cuda(F.normalize(self.dict_of_random_mats[k], p=2, dim=1))
        for k in self.dict_of_linenar_comb:
            self.dict_of_linenar_comb[k] = self.param_cuda(F.normalize(self.dict_of_linenar_comb[k], p=2, dim=1))
            
    def triple_classification(self, pos_h, pos_t, pos_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        #pos_r_e = self.rel_embeddings(pos_r) #\vv{r}
        pos_r_e = self.vvrel_embedding_func(pos_r)
        pos_sub = pos_h_e + pos_r_e - pos_t_e

        num_ent = pos_h_e.shape[0]
        #unique_time = time.time()
        unique_rels_2_unique_proj_idx = self.unique_rels(pos_r)
        #print("unique time:", unique_time-time.time())
        unique_rel_tensor = to_cuda(longTensor([int(key) for key in unique_rels_2_unique_proj_idx]))
        #comb  = time.time()
        #This has shape [#unique_rel x 2 x emb_dim]
        unique_bases = self.comb_mats(unique_rel_tensor) #this is AT, A is unique_bases.t()
        #this will have shape [#unique_rel x 2]
        ####THIS HERE IS WRONG
                #print("comb time :", time.time()-comb)
        #inv_time = time.time()
        ATA_inverse = self.inverse(torch.bmm(unique_bases, unique_bases.transpose(1,2)))
        #print("ATA inverse time :", time.time()-inv_time)
        #proj_time = time.time()
        unique_projmat = to_cuda(torch.cat([torch.eye(self.embedding_size)]*len(unique_rels_2_unique_proj_idx)).view(len(unique_rels_2_unique_proj_idx), self.embedding_size, self.embedding_size)) - torch.bmm(torch.bmm(unique_bases.transpose(1,2), ATA_inverse), unique_bases)
        #print("Multiplying for projection :", time.time()-proj_time)
        #Projection Matrices 
        #Make in the shape of num_ent x emb_dim x emb_dim  (every proj matrix at every num_ent)
        #assign_time = time.time()
        
        unique_proj_idx = [unique_rels_2_unique_proj_idx[pos_r[i].item()] for i in range(num_ent)] 
        P_pos_r = unique_projmat[unique_proj_idx]
        #need to fix here
        bmmed = torch.bmm(P_pos_r, torch.cat(pos_sub).view(num_ent,self.embedding_size,1)).squeeze(2) 
        #print("Time for bmm-ing:", time.time()-bmm_time)

        #print("time for bmm:", time.time() - bmm_tim)
        pos_sub = bmmed[:num_ent]
        
        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_sub), 1)
        else:
            pos = torch.sum((pos_sub) ** 2, 1)
        return pos, neg
    
    def comb_mats_no_which_dataset(self, rel_tensor, in_batch=True, cpu=False):
        stack_mats = to_cuda(torch.zeros(rel_tensor.shape[-1], 2, self.embedding_size))
        i=0
        for rel_tent in rel_tensor:
            rel = rel_tent.item()
            if rel in self.equiv_flip_dict:
                rel = self.equiv_flip_dict[rel]
            p_rel = self.r_id2_vrr[rel][0]
            #no need to stack 
            if p_rel == rel or (self.rank_start[rel] ==self.rank_start[p_rel] and self.rank_end[rel] == self.rank_end[p_rel]):
                if self.rank_end[rel] ==0:
                    result = to_cuda(torch.zeros(2, self.embedding_size))
                    result[0] = self.dict_of_random_mats[str(p_rel)]
                else:
                    result = self.dict_of_random_mats[str(p_rel)]
                if cpu==True:
                    stack_mats[i] = result.cpu()
                stack_mats[i] = result 
            else: 
                #stack everything from above and return 
                result = torch.mm(self.dict_of_linenar_comb[str(rel)], self.dict_of_random_mats[str(p_rel)])
                result_num_rank = result.shape[0]
                if  cpu==True:
                    stack_mats[i][:result_num_rank] = result.cpu()
                stack_mats[i][:result_num_rank] = result
            i+=1
        return stack_mats
    
    def inverse_no_which_dataset(self, tensor_3d):
        inverse_tensor = to_cuda(torch.zeros(tensor_3d.shape[0], tensor_3d.shape[1], tensor_3d.shape[2]))
        for i in range(tensor_3d.shape[0]):
            #assert (tensor_3d[i].shape[0] == 2 and tensor_3d[i].shape[1] == 2)
            a = tensor_3d[i,0,0]; b = tensor_3d[i,0,1]; c = tensor_3d[i,1,0]; d = tensor_3d[i,1,1]
            det = a*d-b*c
            if det !=0:
                #inverse_tensor[i] = torch.inverse(tensor_3d[i])
                #try a new inverse   
                inverse_tensor[i] = to_cuda(torch.tensor([[d, -b], [-c, a]]))*1/(det)
            else:
                #print("inverse", 1/tensor_3d[i][0][0])
                inverse_tensor[i] = to_cuda(torch.eye(tensor_3d.shape[1])) * 1/a
        return inverse_tensor

class TransINTModelTestTransH(nn.Module):
    def __init__(self, config, rank_file, r_id2_vrr, equiv_flip_dict):
        super(TransINTModelTestTransH, self).__init__()
        #there are really just two parameters:
        #self.vvr_bundle_heads, self.dict_of_random_mats 
        #Parameter
        self.vvr_bundle_heads = nn.ParameterDict({str(k): self.param_cuda(v) for k, v in rank_file['vvr_bundle_heads'].items()} )
        self.rank_start = rank_file['rank_start']
        self.rank_end = rank_file['rank_end']
        #Parameter
        self.dict_of_random_mats = nn.ParameterDict({str(k):self.param_cuda(v) for k, v in rank_file['dict_of_random_mats'].items()})
        #self.dict_of_masks = {k: (to_cuda(v[0]), to_cuda(v[1])) for k, v in rank_file['dict_of_masks'].items()}
        self.r_id2_vrr = r_id2_vrr
        self.equiv_flip_dict = equiv_flip_dict
        self.equiv_flip_dict_both_sides = copy.deepcopy(equiv_flip_dict)
        for k, v in self.equiv_flip_dict.items():
            self.equiv_flip_dict_both_sides[v]=k
        
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size
        
        #entity embeddings
        vv_ent_weight = floatTensor(self.entity_total, self.embedding_size)
        nn.init.xavier_uniform_(vv_ent_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(vv_ent_weight)
        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        
        #This part is for TransH
        vv_rel_weight = to_cuda(torch.randn(self.entity_total, self.embedding_size))
        #nn.init.xavier_uniform_(vv_rel_weight)
        self.vvrel_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.vvrel_embeddings.weight = nn.Parameter(vv_rel_weight)
        #normalize_vvrel_emb = F.normalize(self.vvrel_embeddings.weight.data, p=2, dim=1)
        #self.vvrel_embeddings.weight.data = normalize_vvrel_emb
        
        ##Bases: num_rel x embedding_size -1  x embedding_size
# =============================================================================
#         bases_weight = floatTensor(self.relation_total, self.embedding_size -1, self.embedding_size)
#         nn.init.xavier_uniform_(bases_weight)
#         bases_weight = F.normalize(bases_weight, p=2, dim=1)
#         self.bases = nn.Parameter(bases_weight)
# =============================================================================
        bases_weight = floatTensor(self.relation_total, self.embedding_size)
        nn.init.xavier_uniform_(bases_weight)
        bases_weight = F.normalize(bases_weight, p=2, dim=1)
        self.bases = nn.Parameter(bases_weight)

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        #pos_r_e = self.rel_embeddings(pos_r) #\vv{r}
        pos_r_e = self.vvrel_embeddings(pos_r)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e =  self.vvrel_embeddings(neg_r)
        
        num_ent = pos_h_e.shape[0]
        unique_rels_projs = self.unique_rels(torch.cat([pos_r,neg_r]))
        basis_rels = {}
        printed = 0
        for rel in unique_rels_projs:
            if (rel in self.equiv_flip_dict_both_sides) and  (self.equiv_flip_dict_both_sides[rel] in basis_rels):
                unique_rels_projs[rel] =unique_rels_projs[self.equiv_flip_dict_both_sides[rel]]
            else:
                #start_time = time.time()
                basis = self.stack_mats(to_cuda(torch.tensor([rel])))
                #if printed == 0:
                    #print("time for basis:", time.time() - start_time)
                basis_rels[rel] = basis
                proj_time = time.time()
                #print("basis : ", basis[0])
                #print("basis shape: ", basis[0].shape)
                #pickle.dump(basis[0], open("test_basis.p", "wb"))
                unique_rels_projs[rel] = to_cuda(torch.eye(self.embedding_size)) - projection_matrix_transINT(basis)[0]
                #if printed == 0:
                   # print("time for projection:", time.time() - proj_time)
                printed =1
        
        #Projection Matrices 
        #Make in the shape of num_ent x emb_dim x emb_dim  (every proj matrix at every num_ent)
        #P_pos_r = [unique_rels_projs[rel_t.item()] for rel in pos_r_e]
        #assign= time.time()
        P_pos_r = floatTensor(num_ent*3, self.embedding_size, self.embedding_size)
        P_neg_r = floatTensor(num_ent*3, self.embedding_size, self.embedding_size)
        for i in range(num_ent):
            rel = pos_r[i].item()
            P_pos_r[i] = unique_rels_projs[rel]; P_pos_r[i+num_ent] = unique_rels_projs[rel];  P_pos_r[i+2*num_ent] = unique_rels_projs[rel]
            rel_neg = neg_r[i].item()
            P_neg_r[i] = unique_rels_projs[rel]; P_neg_r[i+num_ent] = unique_rels_projs[rel]; P_neg_r[i+2*num_ent] = unique_rels_projs[rel]
        #print("time for assignement:", time.time() - assign)
        #Apply projection to the entity vector embeddngs
        #Make column vectors in the shape of num_ent x emb_dim x 1
        
        #bmm_tim = time.time()
        #Going to have shape num_ent x emb_dim x 1 when came out of the bmm
        pos_bmm = torch.bmm(P_pos_r, torch.cat([pos_h_e, pos_t_e, pos_r_e]).view(3*num_ent,self.embedding_size,1)).squeeze(2) 
        #print("time for bmm:", time.time() - bmm_tim)
        neg_bmm = torch.bmm(P_neg_r, torch.cat([neg_h_e, neg_t_e, neg_r_e]).view(3*num_ent,self.embedding_size,1)).squeeze(2) 

        pos_h_e = pos_bmm[:num_ent]
        pos_t_e = pos_bmm[num_ent:2*num_ent]
        neg_h_e = neg_bmm[:num_ent]
        neg_t_e = neg_bmm[num_ent:2*num_ent]
        pos_r_e = pos_bmm[2*num_ent:3*num_ent]
        neg_r_e = neg_bmm[2*num_ent:3*num_ent]

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg
    
    

    def stack_mats(self, rel_tensor):
        stack_mats = []
        for rel_tent in rel_tensor:
            rel = rel_tent.item()
            stack_mats.append(self.bases[rel].view(1,-1))
        return stack_mats
        
    def vvrel_embedding_func(self, rel_tensors):
        return_tensor = floatTensor(rel_tensors.shape[0], self.embedding_size) #cuda?
        i=0
        for rel_tent in rel_tensors:
            rel = rel_tent.item()
            return_tensor[i] = self.vvr_bundle_heads[str(self.r_id2_vrr[rel][0])] * self.r_id2_vrr[rel][1]
            i+=1
        return return_tensor
    
    def normalize_each_row_rand_mat(self, rel):
        if rel in self.equiv_flip_dict:
            rel = self.equiv_flip_dict[rel]
        p_rel = self.r_id2_vrr[rel][0]
        if p_rel == rel or (self.rank_start[rel] ==self.rank_start[p_rel] and self.rank_end[rel] == self.rank_end[p_rel]):
            self.dict_of_random_mats[str(p_rel)] = self.param_cuda(F.normalize(self.dict_of_random_mats[str(p_rel)], p=2, dim=1))
        else:
            self.dict_of_random_mats[str(rel)] =self.param_cuda(F.normalize(self.dict_of_random_mats[str(rel)], p=2, dim=1))
            self.dict_of_random_mats[str(p_rel)] = self.param_cuda(F.normalize(self.dict_of_random_mats[str(p_rel)], p=2, dim=1))
    
    def vvrel_all_weights(self):
        return_np = np.zeros((self.relation_total, self.embedding_size))
        for rel in range(self.relation_total):
            #return_np[i] = v.data.cpu().numpy()
            return_np[rel] = self.vvrel_embedding_func(to_cuda(torch.tensor([rel]))).data.cpu().numpy()
        return return_np
    
    def param_cuda(self, param):
        return nn.Parameter(to_cuda(param), requires_grad = True)
    
    def unique_rels(self, rel_tensors):
        unique_rels_projs = {}
        for rel_tent in rel_tensors:
            rel = rel_tent.item()
            if not(rel in unique_rels_projs):
                unique_rels_projs[rel] =1
        return unique_rels_projs
    
    def normalize_vvrel_embeddings(self):
        for k in self.vvr_bundle_heads.keys():
            normalized = F.normalize(self.vvr_bundle_heads[k].data, p=2, dim=1)
            self.vvr_bundle_heads[k].data = normalized
