#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:59:35 2019

@author: TiffMin
"""
import pickle
tripleDict = pickle.load(open('datasets/KALE_datasets/fb122/train_triples_list.p', 'rb')) + pickle.load(open('datasets/KALE_datasets/fb122/test_triples_list.p', 'rb')) +pickle.load(open('datasets/KALE_datasets/fb122/tval_triples_list.p', 'rb')) 
#All the dicts in the tripleDict given head and rel only
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


pickle.dump(tripledict_given_head_rel, open('datasets/KALE_datasets/fb122/tripledict_given_head_rel','wb'))
pickle.dump(tripledict_given_tail_rel, open('datasets/KALE_datasets/fb122/tripledict_given_tail_rel','wb'))


#All the possible_imp's given head and rel only (head ,rel, equiv_child_dict):
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
                #added +=1
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
    
                
pickle.dump(possible_imp_given_head_rel, open('datasets/KALE_datasets/fb122/possible_imp_given_head_rel','wb'))
pickle.dump(possible_imp_given_tail_rel, open('datasets/KALE_datasets/fb122/possible_imp_given_tail_rel','wb'))



#no needed                 
added=0
for (h,r) in possible_imp_given_head_rel:
    if r in equiv_child_dict:
        children = [pair[0] for pair in equiv_child_dict[r] if pair[1] == -1]
        for ch_rel in children:
            if (h,ch_rel) in possible_imp_given_tail_rel:
                added +=1
#    
def argwhereHead(head, tail, rel, array, tripleDict, equiv_child_dict=None,impfilter = False):
    abs_rank_of_head = int(np.argwhere(rankArrayHead[0]==head))+1 
    minus =0
    if (tail, rel) in tripledict_given_tail_rel:
        nums = tripledict_given_tail_rel[(tail, rel)]
        for num in nums:
            if int(np.argwhere(rankArrayHead[0]==num))+1  < abs_rank_of_head:
                minus +=1
    return abs_rank_of_head - minus
    
tripledict_given_head_rel={}
tripledict_given_tail_rel={}