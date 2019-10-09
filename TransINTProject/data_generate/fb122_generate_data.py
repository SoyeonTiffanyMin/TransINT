#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:52:38 2019

@author: TiffMin
"""

import random
import numpy as np
#from scipy.stats import ortho_group
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import pickle

##########Script for generating null data
def assign_vvr_to_each_bundlehead(equiv_flip_pairs, total_dimensions):
    vvr_bundle_heads = {}
    for pair in equiv_flip_pairs:
        if pair in [(34,42), (26,58)]:
            pass
        elif pair == (32,32): 
            vvr_bundle_heads[32] = nn.Parameter(F.normalize(torch.zeros(1, total_dimensions), p=2, dim=1), requires_grad = False)
        #randomly sample and add to vvr_list
        #need to do xavier and shit here
        else:
            head = pair[0]
            if head<0:
                save_head = -(head+100)
            else:
                save_head = head
            vvr_bundle_heads[save_head] = nn.Parameter(F.normalize(nn.init.xavier_uniform_(torch.randn(1, total_dimensions)), p=2, dim=1), requires_grad = True)
    return vvr_bundle_heads
 

def NULLassign_rank_to_tree():
    rank_start = {}
    rank_end = {}    
    #
    #then do everything else / assign rank_start to the leaves 
    #gap, center ,var, seed,  total_dimensions
    for h in [3,0]:
        #100 or -103 go first
        if h ==3:
            rank_start[h] = 0
            rank_end[h] = 1
        elif h ==0:
            rank_start[h] = rank_start[3]
            rank_end[h]  = rank_end[3]
        #the other parent go first #don't do this!
        for leaf in direct_child_dict[h]:
            rank_start[leaf] = 0 
            rank_end[leaf] = 0
            #  
    for pair in [pair for pair in equiv_flip_pairs if (not pair in [(3,0), (34,42), (26,58)])]:
        rank_start[pair[0]] = 0
        rank_end[pair[0]] = 0
        rank_start[pair[1]] = 0
        rank_end[pair[1]] = 0
    return rank_start, rank_end

import random

def NULLassign_random_mats(direct_child_dict, vvr_bundle_heads, rank_end):
    dict_of_random_mats = {}  #rel_id: 
    #dict_of_masks = {} not really needed 
    dict_of_linenar_comb = {} 
    dict_of_offset = {} 
    for h in vvr_bundle_heads:
        dict_of_random_mats[h] = nn.Parameter(F.normalize(nn.init.xavier_uniform_(torch.randn(rank_end[h] +1, total_dimensions)), p=2, dim=1), requires_grad = True)
        dict_of_offset[h] = nn.Parameter(F.normalize(nn.init.xavier_uniform_(torch.randn(1, total_dimensions)), p=2, dim=1), requires_grad = True)
    for l in direct_child_dict[3] + [48, 51, 82]:
        dict_of_linenar_comb[l] = nn.Parameter(torch.randn(1,2), requires_grad = True)
# =============================================================================
#     for i in range(122):
#         if not(i in dict_of_linenar_comb):
#             dict_of_linenar_comb[l] = nn.Parameter(torch.ones(1,2), requires_grad = False)       
# =============================================================================
    return dict_of_random_mats, dict_of_linenar_comb, dict_of_offset


###Main Function
def NULLgenerate_data_for_model(total_dimensions):
    rank_start, rank_end = NULLassign_rank_to_tree()
    #2. Now assign \vvr to these 
    #just to all the bundles 
    vvr_bundle_heads = assign_vvr_to_each_bundlehead(equiv_flip_pairs, total_dimensions) 
    #Now do for range(122), {r_id: which_bundle, plus_or_minus} #<- completely Fixed
    #
    #3. Assign rand mats to those in bundles
    #dict_of_random_mats, dict_of_masks = assign_random_mats(rank_start, rank_end, total_dimensions)
    #
    #4. 5. 나머지 positive 애들한테 assign rank, assign vvr, assign rand_mat 
    in_bundle = copy.deepcopy(list(rank_start.keys()))
    for i in range(122):
        if not(i in in_bundle):
            rank_start[i], rank_end[i] = 0, 1
            vvr_bundle_heads[i] = nn.Parameter(F.normalize(nn.init.xavier_uniform_(torch.randn(1, total_dimensions)), p=2, dim=1), requires_grad = True)
            #r_id2_vrr[i] = (i,1)
    dict_of_random_mats, dict_of_linenar_comb, dict_of_offset = NULLassign_random_mats(direct_child_dict, vvr_bundle_heads, rank_end)
    return vvr_bundle_heads, rank_start, rank_end, dict_of_random_mats, dict_of_linenar_comb, dict_of_offset

###############
######Generate NULL data
##############
total_dimensions = 20
vvr_bundle_heads, rank_start, rank_end, dict_of_random_mats, dict_of_linenar_comb, dict_of_offset  = NULLgenerate_data_for_model(total_dimensions)
hr_gap1_dict = {'vvr_bundle_heads':vvr_bundle_heads, 
                'rank_start':rank_start, 
                'rank_end':rank_end, 
                'dict_of_random_mats':dict_of_random_mats, 
                'dict_of_linenar_comb':dict_of_linenar_comb,
                'dict_of_offset':dict_of_offset}
pickle.dump(hr_gap1_dict, open('/data/scratch-oc40/symin95/thesis_kg_project/transint/knowledge_representation_pytorch-master/datasets/KALE_datasets/fb122/NULL_dim_20_hr_gap1_dict.p', 'wb'))
