#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:41:28 2019

@author: TiffMin
"""

#Make tail per head and head per tail for FB122
relations = pickle.load(open('datasets/KALE_datasets/fb122/rel_idx.p', 'rb'))['id2rel_dict']
entities = pickle.load(open('datasets/KALE_datasets/fb122/ent_idx.p', 'rb'))['id2ent_dict']
tripleDict = pickle.load(open('datasets/KALE_datasets/fb122/train_triples_list.p', 'rb')) + pickle.load(open('datasets/KALE_datasets/fb122/test_triples_list.p', 'rb')) +pickle.load(open('datasets/KALE_datasets/fb122/tval_triples_list.p', 'rb'))
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
    
pickle.dump(avg_tail_per_head, open('datasets/KALE_datasets/fb122/tail_per_head.p', 'wb'))
pickle.dump(avg_head_per_tail, open('datasets/KALE_datasets/fb122/head_per_tail.p', 'wb'))