#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 07:47:40 2019

@author: TiffMin
"""

#Nell Dataset Generate 

import pickle
import copy

#First Location 
import os

NL_test_triples_file = open("datasets/NELL_datasets/location/test.txt", "r")
NL_train_triples_file = open("datasets/NELL_datasets/location/train.txt", "r")
#NL_val_triples_file = open("datasets/NELL_datasets/location/valid.txt", "r")

ent2id_dict = {}
rel2id_dict = {}
ent_counter = 0; rel_counter =0
for line in NL_test_triples_file:
    line = line.strip('\n')
    e1, e2, r = line.split('\t')
    if not(e1 in ent2id_dict):
        ent2id_dict[e1] = ent_counter 
        ent_counter +=1 
    if not(e2 in ent2id_dict):
        ent2id_dict[e2] = ent_counter 
        ent_counter +=1
    if not(r in rel2id_dict):
        rel2id_dict[r] = rel_counter
        rel_counter +=1
        
for line in NL_train_triples_file:
    line = line.strip('\n')
    e1, e2,r  = line.split('\t')
    if not(e1 in ent2id_dict):
        ent2id_dict[e1] = ent_counter 
        ent_counter +=1 
    if not(e2 in ent2id_dict):
        ent2id_dict[e2] = ent_counter 
        ent_counter +=1
    if not(r in rel2id_dict):
        rel2id_dict[r] = rel_counter
        rel_counter +=1
        
id2ent_dict = {v:k for k,v in ent2id_dict.items()}
id2rel_dict = {v:k for k,v in rel2id_dict.items()}
NL_test_triples_file = open("datasets/NELL_datasets/location/test.txt", "r")
NL_train_triples_file = open("datasets/NELL_datasets/location/train.txt", "r")

test_triples_list = []
for line in NL_test_triples_file:
    line = line.strip('\n')
    e1, e2,r  = line.split('\t')
    test_triples_list.append((ent2id_dict[e1], ent2id_dict[e2], rel2id_dict[r]))
    
train_triples_list = []
for line in NL_train_triples_file:
    line = line.strip('\n')
    e1, e2, r = line.split('\t')
    train_triples_list.append((ent2id_dict[e1], ent2id_dict[e2], rel2id_dict[r]))


pickle.dump({'ent2id_dict': ent2id_dict, 'id2ent_dict':id2ent_dict}, open("datasets/NELL_datasets/location/ent_idx.p",'wb'))
pickle.dump({'rel2id_dict': rel2id_dict, 'id2rel_dict':id2rel_dict}, open("datasets/NELL_datasets/location/rel_idx.p",'wb'))

#save triples to pickle
pickle.dump(test_triples_list, open("datasets/NELL_datasets/location/test_triples_list.p",'wb'))
pickle.dump(train_triples_list, open("datasets/NELL_datasets/location/train_triples_list.p",'wb'))

###
## Make dicts with revs 
###

id2rel_dict_with_revs = {}
for r_id in id2rel_dict:
    id2rel_dict_with_revs[r_id] = id2rel_dict[r_id]
    reverse_r_id = -(100+r_id)
    id2rel_dict_with_revs[reverse_r_id] = '_REV'+id2rel_dict[r_id]
rel2id_dict_with_revs = {v:k for k,v in id2rel_dict_with_revs.items()}
pickle.dump({'rel2id_dict_with_revs': rel2id_dict_with_revs, 'id2rel_dict_with_revs':id2rel_dict_with_revs}, open("datasets/NELL_datasets/location/rel_idx_with_revs.p",'wb'))

fb122_implication_dict_id2id = {3:[2], 0:[-104]}
pickle.dump(fb122_implication_dict_id2id, open('datasets/NELL_datasets/location/implication_dict_id2id.p', 'wb'))
