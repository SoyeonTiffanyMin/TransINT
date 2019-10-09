o #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:55:52 2019

@author: TiffMin
"""
#WN18
import pickle
import copy
WN18_test_triples_file = open("datasets/KALE_datasets/wn18/wn18_triples.test", "r")
WN18_train_triples_file = open("datasets/KALE_datasets/wn18/wn18_triples.train", "r")
WN18_val_triples_file = open("datasets/KALE_datasets/wn18/wn18_triples.valid", "r")

ent2id_dict = {}
rel2id_dict = {}
ent_counter = 0; rel_counter =0
for line in WN18_test_triples_file:
    line = line.strip('\n')
    e1, r, e2 = line.split('\t')
    if not(e1 in ent2id_dict):
        ent2id_dict[e1] = ent_counter 
        ent_counter +=1 
    if not(e2 in ent2id_dict):
        ent2id_dict[e2] = ent_counter 
        ent_counter +=1
    if not(r in rel2id_dict):
        rel2id_dict[r] = rel_counter
        rel_counter +=1
        
for line in WN18_train_triples_file:
    line = line.strip('\n')
    e1, r, e2 = line.split('\t')
    if not(e1 in ent2id_dict):
        ent2id_dict[e1] = ent_counter 
        ent_counter +=1 
    if not(e2 in ent2id_dict):
        ent2id_dict[e2] = ent_counter 
        ent_counter +=1
    if not(r in rel2id_dict):
        rel2id_dict[r] = rel_counter
        rel_counter +=1

for line in WN18_val_triples_file:
    line = line.strip('\n')
    e1, r, e2 = line.split('\t')
    if not(e1 in ent2id_dict):
        ent2id_dict[e1] = ent_counter 
        ent_counter +=1 
    if not(e2 in ent2id_dict):
        ent2id_dict[e2] = ent_counter 
        ent_counter +=1
    if not(r in rel2id_dict):
        rel2id_dict[r] = rel_counter
        rel_counter +=1

#id2ent, id2rel dicts
id2ent_dict = {v:k for k,v in ent2id_dict.items()}
id2rel_dict = {v:k for k,v in rel2id_dict.items()}

WN18_test_triples_file = open("datasets/KALE_datasets/wn18/wn18_triples.test", "r")
WN18_train_triples_file = open("datasets/KALE_datasets/wn18/wn18_triples.train", "r")
WN18_val_triples_file = open("datasets/KALE_datasets/wn18/wn18_triples.valid", "r")

test_triples_list = []
for line in WN18_test_triples_file:
    line = line.strip('\n')
    e1, r, e2 = line.split('\t')
    test_triples_list.append((ent2id_dict[e1], ent2id_dict[e2], rel2id_dict[r]))
    
train_triples_list = []
for line in WN18_train_triples_file:
    line = line.strip('\n')
    e1, r, e2 = line.split('\t')
    train_triples_list.append((ent2id_dict[e1], ent2id_dict[e2], rel2id_dict[r]))

val_triples_list = []
for line in WN18_val_triples_file:
    line = line.strip('\n')
    e1, r, e2 = line.split('\t')
    val_triples_list.append((ent2id_dict[e1], ent2id_dict[e2], rel2id_dict[r]))

#Save to pickle
pickle.dump({'ent2id_dict': ent2id_dict, 'id2ent_dict':id2ent_dict}, open("datasets/KALE_datasets/wn18/ent_idx.p",'wb'))
pickle.dump({'rel2id_dict': rel2id_dict, 'id2rel_dict':id2rel_dict}, open("datasets/KALE_datasets/wn18/rel_idx.p",'wb'))

#save triples to pickle
pickle.dump(test_triples_list, open("datasets/KALE_datasets/wn18/test_triples_list.p",'wb'))
pickle.dump(train_triples_list, open("datasets/KALE_datasets/wn18/train_triples_list.p",'wb'))
pickle.dump(val_triples_list, open("datasets/KALE_datasets/wn18/tval_triples_list.p",'wb'))


#Make dicts with revs 
id2rel_dict_with_revs = {}
for r_id in id2rel_dict:
    id2rel_dict_with_revs[r_id] = id2rel_dict[r_id]
    reverse_r_id = -(100+r_id)
    id2rel_dict_with_revs[reverse_r_id] = '_REV'+id2rel_dict[r_id]
rel2id_dict_with_revs = {v:k for k,v in id2rel_dict_with_revs.items()}
pickle.dump({'rel2id_dict_with_revs': rel2id_dict_with_revs, 'id2rel_dict_with_revs':id2rel_dict_with_revs}, open("datasets/KALE_datasets/wn18/rel_idx_with_revs.p",'wb'))



#Parse rules 
WN18_rules_file = open("datasets/KALE_datasets/wn18/wn18_rule", "r")

wn18_relation2id_dict = rel2id_dict

fb122_implication_dict_rel2rel = {} #k:v if k->v
fb122_implication_dict_id2id = {}
for line in WN18_rules_file: 
    arrow_idx = line.index('=')
    one_more =0
    if line[-1] == '\n':
        one_more = 1
    rel1 = line[0:arrow_idx-5]; rel2 = line[arrow_idx+3:-(5+one_more)]
    #    
    if line[arrow_idx-5:arrow_idx] == '(x,y)': pass
    elif line[arrow_idx-5:arrow_idx] == '(y,x)': rel1 = '_REV' + rel1
    elif line[arrow_idx-5:arrow_idx] == '(y,z)': pass
    else: raise Exception('weird parsing of rel 1 in line: ', line)
    #
    if line[-(5+one_more):len(line)-one_more] == '(x,y)': pass
    elif line[-(5+one_more):len(line)-one_more] == '(y,x)': rel2= '_REV' + rel2
    elif line[arrow_idx-5:arrow_idx] == '(y,z)': pass
    else: raise Exception('weird parsing of rel 2 in line: ', line)
    #
    rel1_id = rel2id_dict_with_revs[rel1]; rel2_id = rel2id_dict_with_revs[rel2]
    if not(rel1 in fb122_implication_dict_rel2rel):
        fb122_implication_dict_rel2rel[rel1] = []
        fb122_implication_dict_id2id[rel1_id] = []
    fb122_implication_dict_rel2rel[rel1].append(rel2)
    fb122_implication_dict_id2id[rel1_id].append(rel2_id)
    
pickle.dump(fb122_implication_dict_rel2rel, open('datasets/KALE_datasets/wn18/wn18_implication_dict_rel2rel.p', 'wb'))
pickle.dump(fb122_implication_dict_id2id, open('datasets/KALE_datasets/wn18/wn18_implication_dict_id2id.p', 'wb'))
#Ran until here

def all_imp_dict(implication_dict):
    new_implication_dict = copy.deepcopy(implication_dict)
    for k in implication_dict:
        new_implication_dict[k] += implications_depth_n(implication_dict, k, 3)
    return new_implication_dict

def implications_depth1(implication_dict, node):
    if node in implication_dict:
        new_ps =[]
        node_parents = implication_dict[node]
        for p in node_parents:
            if p in implication_dict:
                new_ps += implication_dict[p]
        
        return new_ps
    else:
        return []
            
def implications_depth_n(implication_dict, node, depth):
    if depth==1:
        if node in implication_dict:
            depth1_ps = implications_depth1(implication_dict, node)
        else:
            depth1_ps =[]
        return depth1_ps
    elif depth==2:
        depth1_ps = implications_depth_n(implication_dict, node, 1)
        if node in implication_dict:
            node_parents = implication_dict[node]
            for p in node_parents:
                depth1_ps += implications_depth_n(implication_dict, p, 1)
        return depth1_ps
    else:
        depth1_ps = implications_depth_n(implication_dict, node, 1)
        cur_depth = depth
        while cur_depth !=0:
            if node in implication_dict:
                node_parents = implication_dict[node]; cur_depth = cur_depth-1
                for p in node_parents:
                    depth1_ps += implications_depth_n(implication_dict, p, cur_depth)
            else:
                pass
        return depth1_ps

#wordnet 
#load sysnet_name.json, hyeprnym hd5 
#


complete_fb122_implication_dict_id2id = all_imp_dict(fb122_implication_dict_id2id)
#complete_fb122_implication_dict_rel2rel = all_imp_dict(fb122_implication_dict_rel2rel)
complete_fb122_implication_dict_rel2rel = {}
for k, v in complete_fb122_implication_dict_id2id.items():
    complete_fb122_implication_dict_rel2rel[id2rel_dict_with_revs[k]] = [id2rel_dict_with_revs[vv] for vv in v]

pickle.dump(complete_fb122_implication_dict_id2id, open('datasets/KALE_datasets/wn18/complete_wn18_implication_dict_id2id.p', 'wb'))
pickle.dump(complete_fb122_implication_dict_rel2rel, open('datasets/KALE_datasets/wn18/complete_wn18_implication_dict_rel2rel.p', 'wb'))


    
#Just affect all triples or something? 
    
#implication 2id 
#how to construct a tree? 
    
#First make implication dict 
    
# =============================================================================
# class ImplicationTree:
#   def __init__(self, implication_dict):
#     for k, k_parents in implication_dict.items():
#         
#     
#   def rank(self, node):
#     print("Hello my name is " + abc.name)
# 
# 
#   @property
#   def get_head(self):
#       return None
#   
# =============================================================================
# =============================================================================
# for k, k_parents in implication_dict.items():
#     for p in k_parents:
#         p1 = copy.deepcopy(p)
#         while p1 in implication_dict:
#             implication_dict[k].append(p1)
#         
# =============================================================================
