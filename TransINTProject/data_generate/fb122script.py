#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:23:43 2019

@author: TiffMin
"""

#First map each fb122 entity to id, and rel to id 
fb122_test_triples_file = open("datasets/KALE_datasets/fb122/fb122_triples.test", "r")
fb122_train_triples_file = open("datasets/KALE_datasets/fb122/fb122_triples.train", "r")
fb122_val_triples_file = open("datasets/KALE_datasets/fb122/fb122_triples.valid", "r")

ent2id_dict = {}
rel2id_dict = {}
ent_counter = 0; rel_counter =0
for line in fb122_test_triples_file:
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
        
for line in fb122_train_triples_file:
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

for line in fb122_val_triples_file:
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

#Convert all the triples to id's
fb122_test_triples_file = open("fb122_triples.test", "r")
fb122_train_triples_file = open("fb122_triples.train", "r")
fb122_val_triples_file = open("fb122_triples.valid", "r")

#In the order of e1, e2, r 
test_triples_list = []
for line in fb122_test_triples_file:
    line = line.strip('\n')
    e1, r, e2 = line.split('\t')
    test_triples_list.append((ent2id_dict[e1], ent2id_dict[e2], rel2id_dict[r]))
    
train_triples_list = []
for line in fb122_train_triples_file:
    line = line.strip('\n')
    e1, r, e2 = line.split('\t')
    train_triples_list.append((ent2id_dict[e1], ent2id_dict[e2], rel2id_dict[r]))

val_triples_list = []
for line in fb122_val_triples_file:
    line = line.strip('\n')
    e1, r, e2 = line.split('\t')
    val_triples_list.append((ent2id_dict[e1], ent2id_dict[e2], rel2id_dict[r]))

#Save to pickle
pickle.dump({'ent2id_dict': ent2id_dict, 'id2ent_dict':id2ent_dict}, open("ent_idx.p",'wb'))
pickle.dump({'rel2id_dict': rel2id_dict, 'id2rel_dict':id2rel_dict}, open("rel_idx.p",'wb'))

#save triples to pickle
pickle.dump(test_triples_list, open("test_triples_list.p",'wb'))
pickle.dump(train_triples_list, open("train_triples_list.p",'wb'))
pickle.dump(val_triples_list, open("tval_triples_list.p",'wb'))



###Make relations with reverse dicts 
id2rel_dict_with_revs = {}
for r_id in id2rel_dict:
    id2rel_dict_with_revs[r_id] = id2rel_dict[r_id]
    reverse_r_id = -(100+r_id)
    id2rel_dict_with_revs[reverse_r_id] = '_REV'+id2rel_dict[r_id]
rel2id_dict_with_revs = {v:k for k,v in id2rel_dict_with_revs.items()}
pickle.dump({'rel2id_dict_with_revs': rel2id_dict_with_revs, 'id2rel_dict_with_revs':id2rel_dict_with_revs}, open("datasets/KALE_datasets/fb122/rel_idx_with_revs.p",'wb'))


#Now Do Rules
fb122_rules_file = open("datasets/rules/fb122_rule.txt", "r")

fb122_implication_dict_rel2rel = {} #k:v if k->v
fb122_implication_dict_id2id = {}
for line in fb122_rules_file: 
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
    
pickle.dump(fb122_implication_dict_rel2rel, open('datasets/KALE_datasets/fb122/fb122_implication_dict_rel2rel.p', 'wb'))
pickle.dump(fb122_implication_dict_id2id, open('datasets/KALE_datasets/fb122/fb122_implication_dict_id2id.p', 'wb'))


#Now make dict of all implications
def all_imp_dict(implication_dict):
    new_implication_dict = copy.deepcopy(implication_dict)
    for k in implication_dict:
        new_implication_dict[k] += implications_depth_n(implication_dict, node, 3)
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
                
#Let's save
complete_fb122_implication_dict_id2id = all_imp_dict(fb122_implication_dict_id2id)
#complete_fb122_implication_dict_rel2rel = all_imp_dict(fb122_implication_dict_rel2rel)
complete_fb122_implication_dict_rel2rel = {}
for k, v in complete_fb122_implication_dict_id2id.items():
    complete_fb122_implication_dict_rel2rel[id2rel_dict_with_revs[k]] = [id2rel_dict_with_revs[vv] for vv in v]

pickle.dump(complete_fb122_implication_dict_id2id, open('datasets/KALE_datasets/fb122/complete_fb122_implication_dict_id2id.p', 'wb'))
pickle.dump(complete_fb122_implication_dict_rel2rel, open('datasets/KALE_datasets/fb122/complete_fb122_implication_dict_rel2rel.p', 'wb'))

#make chains

fb122_implication_dict_rel2rel = pickle.load(open('datasets/KALE_datasets/fb122/fb122_implication_dict_rel2rel.p', 'rb'))
fb122_implication_dict_id2id = pickle.load( open('datasets/KALE_datasets/fb122/fb122_implication_dict_id2id.p', 'rb'))
id2rel_dict_with_revs = pickle.load(open("datasets/KALE_datasets/fb122/rel_idx_with_revs.p",'rb'))['id2rel_dict_with_revs']

#Make chains
implication_dict = fb122_implication_dict_id2id
list_of_chains = []
for node in implication_dict:
    parents = implication_dict[node]
    curr_chain = [node]
    for p in parents:
        if p in implication_dict:
            list_of_chains.append(implication_dict[p] + [p] + curr_chain)
        else:
            list_of_chains.append([p] + curr_chain)

for rel_id in id2rel_dict_with_revs:
    if not(rel_id in implication_dict):
        list_of_chains.append([rel_id])
        
pickle.dump(list_of_chains, open('datasets/KALE_datasets/fb122/fb122_list_of_all_chains.p', 'wb'))
