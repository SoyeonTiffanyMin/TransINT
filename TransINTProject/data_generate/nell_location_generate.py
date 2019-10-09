#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 08:53:08 2019

@author: TiffMin
"""

r_id2_vrr = {0: (4,-1), 1:(1,1), 2: (2, 1), 3:(2,1), 4:(4,1)}
pickle.dump(r_id2_vrr, open('datasets/NELL_datasets/location/r_id2_vrr.p','wb'))

tripleDict = {pair:1 for pair in test_triples_list + train_triples_list}
pickle.dump(tripleDict, open('datasets/NELL_datasets/location/tripleDict.p','wb'))

entities = id2ent_dict
relations = id2rel_dict

import copy
epe_dict = {}
for ent1 in entities:
    epe_dict[ent1] = 0
tail_per_head = {rel:copy.deepcopy(epe_dict) for rel in relations} #the keys are head
head_per_tail = {rel:copy.deepcopy(epe_dict) for rel in relations} #the keys are tail

for (h,t,r) in tripleDict:
    tail_per_head[r][h] +=1
    head_per_tail[r][t] +=1

r_count = {rel:0 for rel in relations}
for (h,t,r) in tripleDict:
    r_count[r] +=1 
    
avg_tail_per_head = {rel:0 for rel in relations}
avg_head_per_tail = {rel:0 for rel in relations}

for r in relations:
    avg_t=0; avg_h = 0
    nonezro_h = 0; nonezro_t = 0;
    for h in tail_per_head[r]:
        avg_t += tail_per_head[r][h]
        if tail_per_head[r][h]>0:
            nonezro_h +=1
    for t in head_per_tail[r]:
        avg_h += head_per_tail[r][t]
        if head_per_tail[r][t]>0:
            nonezro_t +=1
    avg_tail_per_head[r] =  avg_t/ nonezro_h
    avg_head_per_tail[r] =  avg_h/ nonezro_t

pickle.dump(avg_tail_per_head, open('datasets/NELL_datasets/location/tail_per_head.p', 'wb'))
pickle.dump(avg_head_per_tail, open('datasets/NELL_datasets/location/head_per_tail.p', 'wb'))


##########
###### Triple Dict Given H,T and Possible Triple Dicts 
#########
tripledict_given_head_rel={}
tripledict_given_tail_rel={}

for (h,t,r) in tripleDict:
    if not((h,r) in tripledict_given_head_rel):
        tripledict_given_head_rel[(h,r)] = []
    if not((t,r) in tripledict_given_tail_rel):
        tripledict_given_tail_rel[(t,r)] = []
    tripledict_given_head_rel[(h,r)].append(t)
    tripledict_given_tail_rel[(t,r)].append(h)
    
for k in tripledict_given_head_rel:
    tripledict_given_head_rel[k] = list(set(tripledict_given_head_rel[k]))

for k in tripledict_given_tail_rel:
    tripledict_given_tail_rel[k] = list(set(tripledict_given_tail_rel[k]))


pickle.dump(tripledict_given_head_rel, open('datasets/NELL_datasets/location/tripledict_given_head_rel','wb'))
pickle.dump(tripledict_given_tail_rel, open('datasets/NELL_datasets/location/tripledict_given_tail_rel','wb'))


#######
#Make Possible Triple Dicts
#First make equiv_child_dict
equiv_child_dict = {4:[(0,-1)], 2:[(3,1)]}

    
possible_imp_given_head_rel = {}
possible_imp_given_tail_rel = {}

added =0
for (h,r) in tripledict_given_head_rel:
    possible_imp_given_head_rel[(h,r)] = tripledict_given_head_rel[(h,r)]
    #Now look at other t's that meet the chidl relation of "r"
    if r in equiv_child_dict:
        #print(r ," is in equivchild dict!")
        children = [pair[0] for pair in equiv_child_dict[r] if pair[1] == 1]
        for ch_rel in children:
            if (h,ch_rel) in tripledict_given_head_rel:
                added +=1
                possible_imp_given_head_rel[(h,r)] += tripledict_given_head_rel[(h, ch_rel)]
        children = [pair[0] for pair in equiv_child_dict[r] if pair[1] == -1]
        for ch_rel in children:
            #print((h, ch_rel))
            if (h,ch_rel) in tripledict_given_tail_rel:
                added +=1
                possible_imp_given_head_rel[(h,r)] += tripledict_given_tail_rel[(h, ch_rel)]


for (t,r) in tripledict_given_tail_rel:
    possible_imp_given_tail_rel[(t,r)] = tripledict_given_tail_rel[(t,r)]
    #Now look at other t's that meet the chidl relation of "r"
    if r in equiv_child_dict:
        #print(r ," is in equivchild dict!")
        children = [pair[0] for pair in equiv_child_dict[r] if pair[1] == 1]
        for ch_rel in children:
            if (t,ch_rel) in tripledict_given_tail_rel:
                added +=0
                possible_imp_given_tail_rel[(t,r)] += tripledict_given_tail_rel[(t, ch_rel)]
        children = [pair[0] for pair in equiv_child_dict[r] if pair[1] == -1]
        for ch_rel in children:
            if (t,ch_rel) in tripledict_given_head_rel:
                added +=1
                possible_imp_given_tail_rel[(t,r)] += tripledict_given_head_rel[(t, ch_rel)]

for k in possible_imp_given_head_rel:
    possible_imp_given_head_rel[k] = list(set(possible_imp_given_head_rel[k]))

for k in possible_imp_given_tail_rel:
    possible_imp_given_tail_rel[k] = list(set(possible_imp_given_tail_rel[k]))
    
                
pickle.dump(possible_imp_given_head_rel, open('datasets/NELL_datasets/location/possible_imp_given_head_rel','wb'))
pickle.dump(possible_imp_given_tail_rel, open('datasets/NELL_datasets/location/possible_imp_given_tail_rel','wb'))

##########
###### Generate Data 
#########
import random
import numpy as np
#from scipy.stats import ortho_group
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import pickle

def assign_vvr_to_each_bundlehead(total_dimensions):
    vvr_bundle_heads = {}
    for r in r_id2_vrr:
        if (r in [1,2,4]):
            vvr_bundle_heads[r] = nn.Parameter(F.normalize(nn.init.xavier_uniform_(torch.randn(1, total_dimensions)), p=2, dim=1), requires_grad = True)
    return vvr_bundle_heads
 

def NULLassign_rank_to_tree():
    rank_start = {}
    rank_end = {}    
    #
    #then do everything else / assign rank_start to the leaves 
    #gap, center ,var, seed,  total_dimensions
    for r in [0,2,3,4]:
        if r in [2,4]: #the more general concept
            rank_start[r] = 0
            rank_end[r] = 1
        else:
            rank_start[r] = 0
            rank_end[r] = 0
    return rank_start, rank_end

import random

def NULLassign_random_mats(vvr_bundle_heads, rank_end):
    dict_of_random_mats = {}  #rel_id: 
    #dict_of_masks = {} not really needed 
    dict_of_linenar_comb = {} 
    dict_of_offset = {} 
    for h in vvr_bundle_heads:
        dict_of_random_mats[h] = nn.Parameter(F.normalize(nn.init.xavier_uniform_(torch.randn(rank_end[h] +1, total_dimensions)), p=2, dim=1), requires_grad = True)
        dict_of_offset[h] = nn.Parameter(F.normalize(nn.init.xavier_uniform_(torch.randn(1, total_dimensions)), p=2, dim=1), requires_grad = True)
    for l in [3,0]: #less general ones
        dict_of_linenar_comb[l] = nn.Parameter(torch.randn(1,2), requires_grad = True)
    return dict_of_random_mats, dict_of_linenar_comb, dict_of_offset


###Main Function
def NULLgenerate_data_for_model(total_dimensions):
    rank_start, rank_end = NULLassign_rank_to_tree()
    #2. Now assign \vvr to these 
    #just to all the bundles 
    vvr_bundle_heads = assign_vvr_to_each_bundlehead(total_dimensions) 
    #Now do for range(122), {r_id: which_bundle, plus_or_minus} #<- completely Fixed
    #
    #3. Assign rand mats to those in bundles
    #dict_of_random_mats, dict_of_masks = assign_random_mats(rank_start, rank_end, total_dimensions)
    #
    #4. 5. 나머지 positive 애들한테 assign rank, assign vvr, assign rand_mat 
    in_bundle = copy.deepcopy(list(rank_start.keys()))
    for i in range(5):
        if not(i in in_bundle):
            #rank_start[i], rank_end[i] = 0, 1
            rank_start[i], rank_end[i] = 0, 0
            vvr_bundle_heads[i] = nn.Parameter(F.normalize(nn.init.xavier_uniform_(torch.randn(1, total_dimensions)), p=2, dim=1), requires_grad = True)
            #r_id2_vrr[i] = (i,1)
    dict_of_random_mats, dict_of_linenar_comb, dict_of_offset = NULLassign_random_mats(vvr_bundle_heads, rank_end)
    return vvr_bundle_heads, rank_start, rank_end, dict_of_random_mats, dict_of_linenar_comb, dict_of_offset

total_dimensions = 50
vvr_bundle_heads, rank_start, rank_end, dict_of_random_mats, dict_of_linenar_comb, dict_of_offset  = NULLgenerate_data_for_model(total_dimensions)
hr_gap1_dict = {'vvr_bundle_heads':vvr_bundle_heads, 
                'rank_start':rank_start, 
                'rank_end':rank_end, 
                'dict_of_random_mats':dict_of_random_mats, 
                'dict_of_linenar_comb':dict_of_linenar_comb,
                'dict_of_offset':dict_of_offset}
pickle.dump(hr_gap1_dict, open('/data/scratch-oc40/symin95/thesis_kg_project/transint/knowledge_representation_pytorch-master/datasets/NELL_datasets/location/NULL_dim_50_hr_gap1_dict.p', 'wb'))
