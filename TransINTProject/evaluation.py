#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-15 16:03:42
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os
import numpy as np
import time
import datetime
import random
import multiprocessing
import math
import operator

from itertools import groupby
import pickle

from utils import Triple, getRel, implication_correct, to_cuda

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics.pairwise import pairwise_distances

from projection import *

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor

def isHit10(triple, tree, cal_embedding, tripleDict, isTail):
    # If isTail == True, evaluate the prediction of tail entity
    if isTail == True:
        k = 0
        wrongCount = 0
        while wrongCount < 10:
            k += 15
            tail_dist, tail_ind = tree.query(cal_embedding, k=k)
            for elem in tail_ind[0][k - 15: k]:
                if triple.t == elem:
                    return True
                elif (triple.h, elem, triple.r) in tripleDict:
                    continue
                else:
                    wrongCount += 1
                    if wrongCount > 9:
                        return False
    # If isTail == False, evaluate the prediction of head entity
    else:
        k = 0
        wrongCount = 0
        while wrongCount < 10:
            k += 15
            head_dist, head_ind = tree.query(cal_embedding, k=k)
            for elem in head_ind[0][k - 15: k]:
                if triple.h == elem:
                    return True
                elif (elem, triple.t, triple.r) in tripleDict:
                    continue
                else:
                    wrongCount += 1
                    if wrongCount > 9:
                        return False

# Find the rank of ground truth tail in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereTail(num_ent, headList, tailList, relList, arg_ranks, tripleDict, tripledict_given_head_rel, impfilter=False):
    abs_rank_of_tail =  (arg_ranks == to_cuda(torch.cat([torch.tensor([tailList]).t()]*num_ent, dim=1))).nonzero()[:,1]
    abs_rank_of_tail = to_cuda(abs_rank_of_tail.type(torch.FloatTensor))
    tails_that_exist = []
    #print("head: ", headList[0])
    #print("rel :", relList[0])
    #print("tripledict_given_head_rel:", tripledict_given_head_rel)
    for head, rel in zip(headList, relList):
        if (head, rel) in tripledict_given_head_rel:
            tails_that_exist.append(tripledict_given_head_rel[(head, rel)] )
        else:
            tails_that_exist.append([] )
    i =0
    ranks_to_exclue = []
    #arg_ranks has shape #rel_list x num_ent
    for head, rel in zip(headList, relList):
        correct_tails = tails_that_exist[i]
        ct_ranks = []
        for ct in correct_tails:
            ct_rank = (arg_ranks[i] == ct).nonzero().item()
            if ct_rank <  abs_rank_of_tail[i]:
                ct_ranks.append(ct_rank)
        ranks_to_exclue.append(len(ct_ranks))
        i+=1
    
    return abs_rank_of_tail - to_cuda(torch.tensor(ranks_to_exclue).type(torch.FloatTensor))


# Find the rank of ground truth head in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereHead(num_ent, headList, tailList, relList, arg_ranks, tripleDict, tripledict_given_tail_rel,impfilter = False):
    abs_rank_of_head = (arg_ranks == to_cuda(torch.cat([torch.tensor([headList]).t()]*num_ent, dim=1))).nonzero()[:,1]
    abs_rank_of_head = to_cuda(abs_rank_of_head.type(torch.FloatTensor))
    heads_that_exist = []
    for tail, rel in zip(tailList, relList):
        if (tail, rel) in tripledict_given_tail_rel:
            heads_that_exist.append(tripledict_given_tail_rel[(tail, rel)] )
        else:
            heads_that_exist.append([] )
        
    i =0
    ranks_to_exclue = []
    #arg_ranks has shape #rel_list x num_ent
    for tail, rel in zip(tailList, relList):
        correct_heads = heads_that_exist[i]
        ch_ranks = []
        for ch in correct_heads:
            ch_rank = (arg_ranks[i] == ch).nonzero().item()
            if ch_rank < abs_rank_of_head[i]:
                ch_ranks.append(ch_rank)
        ranks_to_exclue.append(len(ch_ranks))
        i+=1
    return abs_rank_of_head - to_cuda(torch.tensor(ranks_to_exclue).type(torch.FloatTensor))

def pairwise_L1_distances(A, B):
    dist = torch.sum(torch.abs(A.unsqueeze(1) - B.unsqueeze(0)), dim=2)
    return dist

def pairwise_L2_distances(A, B):
    AA = torch.sum(A ** 2, dim=1).unsqueeze(1)
    BB = torch.sum(B ** 2, dim=1).unsqueeze(0)
    dist = torch.mm(A, torch.transpose(B, 0, 1))
    dist *= -2
    dist += AA
    dist += BB
    return dist

def evaluation_transE_helper(testList, tripleDict, ent_embeddings, 
    rel_embeddings, L1_flag, filter, head=0):
    # embeddings are numpy like

    headList = [triple.h for triple in testList]
    tailList = [triple.t for triple in testList]
    relList = [triple.r for triple in testList]

    h_e = ent_embeddings[headList]
    t_e = ent_embeddings[tailList]
    r_e = rel_embeddings[relList]

    # Evaluate the prediction of only head entities
    if head == 1:
        c_h_e = t_e - r_e

        if L1_flag == True:
            dist = pairwise_distances(c_h_e, ent_embeddings, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, ent_embeddings, metric='euclidean')

        rankArrayHead = np.argsort(dist, axis=1)
        # Don't check whether it is false negative
        if filter == False:
            rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
        # Check whether it is false negative
        else:
            rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank = sum(rankListHead)
        hit10Count = len(isHit10ListHead)
        tripleCount = len(rankListHead)

    # Evaluate the prediction of only tail entities
    elif head == 2:
        c_t_e = h_e + r_e

        if L1_flag == True:
            dist = pairwise_distances(c_t_e, ent_embeddings, metric='manhattan')
        else:
            dist = pairwise_distances(c_t_e, ent_embeddings, metric='euclidean')

        rankArrayTail = np.argsort(dist, axis=1)
        if filter == False:
            rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
        else:
            rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayTail)]

        isHit10ListTail = [x for x in rankListTail if x < 10]

        totalRank = sum(rankListTail)
        hit10Count = len(isHit10ListTail)
        tripleCount = len(rankListTail)

    # Evaluate the prediction of both head and tail entities
    else:
        c_t_e = h_e + r_e
        c_h_e = t_e - r_e

        if L1_flag == True:
            dist = pairwise_distances(c_t_e, ent_embeddings, metric='manhattan')
        else:
            dist = pairwise_distances(c_t_e, ent_embeddings, metric='euclidean')

        rankArrayTail = np.argsort(dist, axis=1)
        if filter == False:
            rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
        else:
            rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayTail)]

        isHit10ListTail = [x for x in rankListTail if x < 10]

        if L1_flag == True:
            dist = pairwise_distances(c_h_e, ent_embeddings, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, ent_embeddings, metric='euclidean')

        rankArrayHead = np.argsort(dist, axis=1)
        if filter == False:
            rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
        else:
            rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank = sum(rankListTail) + sum(rankListHead)
        hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
        tripleCount = len(rankListTail) + len(rankListHead)

    return hit10Count, totalRank, tripleCount

class MyProcessTransE(multiprocessing.Process):
    def __init__(self, L, tripleDict, ent_embeddings, 
        rel_embeddings, L1_flag, filter, queue=None, head=0):
        super(MyProcessTransE, self).__init__()
        self.L = L
        self.queue = queue
        self.tripleDict = tripleDict
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        self.L1_flag = L1_flag
        self.filter = filter
        self.head = head

    def run(self):
        while True:
            testList = self.queue.get()
            try:
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
                    self.L1_flag, self.filter, self.L, self.head)
            except:
                time.sleep(5)
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
                    self.L1_flag, self.filter, self.L, self.head)
            self.queue.task_done()

    def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
        L1_flag, filter, L, head):

        hit10Count, totalRank, tripleCount = evaluation_transE_helper(testList, tripleDict, ent_embeddings, 
            rel_embeddings, L1_flag, filter, head)

        L.append((hit10Count, totalRank, tripleCount))

# Use multiprocessing to speed up evaluation
def evaluation_transE(testList, tripleDict, ent_embeddings, rel_embeddings, 
    L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0): #k is config.batch_size
    # embeddings are numpy like

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    # Split the testList into #num_processes parts
    len_split = math.ceil(len(testList) / num_processes)
    testListSplit = [testList[i : i + len_split] for i in range(0, len(testList), len_split)]

    with multiprocessing.Manager() as manager:
        # Create a public writable list to store the result
        L = manager.list()
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = MyProcessTransE(L, tripleDict, ent_embeddings, rel_embeddings,
                L1_flag, filter, queue=queue, head=head)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for subList in testListSplit:
            queue.put(subList)

        queue.join()

        resultList = list(L)

        # Terminate the worker after execution, to avoid memory leaking
        for worker in workerList:
            worker.terminate()

    if head == 1 or head == 2:
        hit10 = sum([elem[0] for elem in resultList]) / len(testList)
        meanrank = sum([elem[1] for elem in resultList]) / len(testList)
    else:
        hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
        meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

    print('Meanrank: %.6f' % meanrank)
    print('Hit@10: %.6f' % hit10)

    return hit10, meanrank        

def evaluation_transINT_helper(testList, tripleDict, ent_embeddings,
    rel_embeddings, model, L1_flag, filter, tripledict_given_tail_rel, tripledict_given_head_rel, k=0, head=0, impfilter=False):
    # embeddings are torch tensor like (No Variable!)
    # Only one kind of relation
    num_ent = ent_embeddings.shape[0]
    num_rel = rel_embeddings.shape[0]
    embedding_size = ent_embeddings.shape[1]
    #offset_lin_combs = model.offset_lin_combs.data #shape: [vvr_bundle_heads_embs's keys x 2]
    
    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    headList = [triple.h for triple in testList]
    tailList = [triple.t for triple in testList]
    relList = [triple.r for triple in testList]


    #r_bases = {rel: model.comb_mats(to_cuda(torch.tensor([rel]))) for rel in relList}
    
    unique_rel_tensor = longTensor(range(num_rel))
    unique_bases = model.comb_mats(unique_rel_tensor) #this is AT, A is unique_bases.t()
    
    #offset_access_index = [model.vvr_bundle_heads_keys[model.r_id2_vrr[rel.item()][0]] for rel in unique_rel_tensor]
    #mult = torch.tensor([model.r_id2_vrr[rel.item()][1] for rel in unique_rel_tensor]).type(torch.FloatTensor).cuda() #shape: [unique_rel]
    #offset_lin_combs_for_unique_rels = model.offset_lin_combs[offset_access_index] 
    #unique_offset = torch.bmm(offset_lin_combs_for_unique_rels.unsqueeze(1), unique_bases).squeeze(1)
    #unique_offset = unique_offset * torch.cat([mult.view(-1,1)] *embedding_size, dim=1)
    
    #unique_offset = torch.bmm(offset_lin_combs_for_unique_rels.unsqueeze(1), unique_bases).squeeze(1)
    ATA_inverse =  model.inverse(torch.bmm(unique_bases, unique_bases.transpose(1,2)))
    unique_projmat =to_cuda(torch.cat([torch.eye(embedding_size)]*num_rel).view(num_rel, embedding_size, embedding_size)) - torch.bmm(torch.bmm(unique_bases.transpose(1,2), ATA_inverse), unique_bases).type(floatTensor) #num_rel x emb x emb
    rel_embeddings_column_pytorch = to_cuda(torch.tensor(rel_embeddings)).view(num_rel,embedding_size,1).type(floatTensor) #originally: embed_dim x num_ent
    proj_all_r = torch.bmm(unique_projmat, rel_embeddings_column_pytorch).squeeze(2)#num_rel x emb
    
    del ATA_inverse
    del rel_embeddings_column_pytorch
    del unique_bases
    
    #To the first "num_ent"'s, project with the first relation
    #To the first "num_ent"'s, project with the second relation
    ent_embeddings_column_pytorch = to_cuda(torch.tensor(np.tile(ent_embeddings,(num_rel, 1)))).view(num_ent*num_rel,embedding_size,1).type(floatTensor) #originally: embed_dim x num_ent
    proj_all_e = to_cuda(torch.zeros(num_ent* num_rel, embedding_size))
    if num_ent < 2000:
        P_pos_r = to_cuda(torch.zeros(num_ent*num_rel, embedding_size, embedding_size))
        for i in range(num_rel):
            P_pos_r[num_ent*i:num_ent*(i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[:num_ent*num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[:num_ent*num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size

    
    elif embedding_size < 100:
        fourth_num_rel = math.ceil(num_rel/4)
        P_pos_r = to_cuda(torch.zeros(num_ent*fourth_num_rel, embedding_size, embedding_size))
        for i in range(fourth_num_rel):
            #print("invalid?: ", torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent).shape)
            #print("invalid pos?: ",  P_pos_r[num_ent*i:num_ent*(i+1)].shape)
            P_pos_r[num_ent*i:num_ent*(i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[:num_ent*fourth_num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[:num_ent*fourth_num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
        
        for i in range(fourth_num_rel, 2*fourth_num_rel):
            #print("pos shape:", P_pos_r[num_ent*i:num_ent*(i+1)].shape)
            pos_i = i - fourth_num_rel
            P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[num_ent*fourth_num_rel:num_ent*2*fourth_num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[num_ent*fourth_num_rel:num_ent*2*fourth_num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
    
        for i in range(2*fourth_num_rel, 3*fourth_num_rel):
            #print("pos shape:", P_pos_r[num_ent*i:num_ent*(i+1)].shape)
            pos_i = i - 2*fourth_num_rel
            P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[num_ent*2*fourth_num_rel:num_ent*3*fourth_num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[num_ent*2*fourth_num_rel:num_ent*3*fourth_num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
        
    
        other_fourth_num_rel = num_rel - 3*fourth_num_rel
        #P_pos_r = floatTensor(num_ent*other_third_num_rel, embedding_size, embedding_size)
        for i in range(3*fourth_num_rel, num_rel):
            pos_i = i - 3*fourth_num_rel
            P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[num_ent*3*fourth_num_rel:, :] = torch.bmm(P_pos_r[:num_ent*other_fourth_num_rel,:,:], ent_embeddings_column_pytorch[num_ent*3*fourth_num_rel:num_ent*num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
    
    
    elif num_ent>40000:
        tenth_num_rel = math.ceil(num_rel/10)
        P_pos_r = to_cuda(torch.zeros(num_ent*tenth_num_rel, embedding_size, embedding_size))
        for j in range(9):
            for i in range(j*tenth_num_rel, (j+1)*tenth_num_rel):
                #print("invalid?: ", torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent).shape)
                #print("invalid pos?: ",  P_pos_r[num_ent*i:num_ent*(i+1)].shape)
                pos_i = i - j*tenth_num_rel
                P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
            proj_all_e[j*num_ent*tenth_num_rel:(j+1)*num_ent*tenth_num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[:num_ent*tenth_num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
            
        other_tenth_num_rel = num_rel - 9*tenth_num_rel
        #P_pos_r = floatTensor(num_ent*other_third_num_rel, embedding_size, embedding_size)
        for i in range(9*tenth_num_rel, num_rel):
            pos_i = i - 9*tenth_num_rel
            P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[num_ent*9*tenth_num_rel:, :] = torch.bmm(P_pos_r[:num_ent*other_tenth_num_rel,:,:], ent_embeddings_column_pytorch[num_ent*9*tenth_num_rel:num_ent*num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
    
    
    else:
        sixth_num_rel = math.ceil(num_rel/6)
        P_pos_r = to_cuda(torch.zeros(num_ent*sixth_num_rel, embedding_size, embedding_size))
        for i in range(sixth_num_rel):
            #print("invalid?: ", torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent).shape)
            #print("invalid pos?: ",  P_pos_r[num_ent*i:num_ent*(i+1)].shape)
            P_pos_r[num_ent*i:num_ent*(i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[:num_ent*sixth_num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[:num_ent*sixth_num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
        
        for i in range(sixth_num_rel, 2*sixth_num_rel):
            #print("pos shape:", P_pos_r[num_ent*i:num_ent*(i+1)].shape)
            pos_i = i - sixth_num_rel
            P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[num_ent*sixth_num_rel:num_ent*2*sixth_num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[num_ent*sixth_num_rel:num_ent*2*sixth_num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
    
        for i in range(2*sixth_num_rel, 3*sixth_num_rel):
            #print("pos shape:", P_pos_r[num_ent*i:num_ent*(i+1)].shape)
            pos_i = i - 2*sixth_num_rel
            P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[num_ent*2*sixth_num_rel:num_ent*3*sixth_num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[num_ent*2*sixth_num_rel:num_ent*3*sixth_num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
        
        for i in range(3*sixth_num_rel, 4*sixth_num_rel):
            #print("pos shape:", P_pos_r[num_ent*i:num_ent*(i+1)].shape)
            pos_i = i - 3*sixth_num_rel
            P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[num_ent*3*sixth_num_rel:num_ent*4*sixth_num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[num_ent*3*sixth_num_rel:num_ent*4*sixth_num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size

        for i in range(4*sixth_num_rel, 5*sixth_num_rel):
            #print("pos shape:", P_pos_r[num_ent*i:num_ent*(i+1)].shape)
            pos_i = i - 4*sixth_num_rel
            P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[num_ent*4*sixth_num_rel:num_ent*5*sixth_num_rel, :] = torch.bmm(P_pos_r, ent_embeddings_column_pytorch[num_ent*4*sixth_num_rel:num_ent*5*sixth_num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size
    
        other_sixth_num_rel = num_rel - 5*sixth_num_rel
        #P_pos_r = floatTensor(num_ent*other_third_num_rel, embedding_size, embedding_size)
        for i in range(5*sixth_num_rel, num_rel):
            pos_i = i - 5*sixth_num_rel
            P_pos_r[num_ent*pos_i:num_ent*(pos_i+1)] = torch.cat([unique_projmat[i].unsqueeze(0)]*num_ent)
        proj_all_e[num_ent*5*sixth_num_rel:, :] = torch.bmm(P_pos_r[:num_ent*other_sixth_num_rel,:,:], ent_embeddings_column_pytorch[num_ent*5*sixth_num_rel:num_ent*num_rel,:,:]).squeeze(2)  #will be num_ent*num_relx self.embedding_size

    
    del ent_embeddings_column_pytorch
    del P_pos_r
    
    #headListE = proj_all_e[headList]
    #tailListE = proj_all_e[tailList]
    relListE = proj_all_r[relList]
    #offListE = unique_offset[relList]
    #print("offlistE:", offListE)
    #print("proj_all_e:", proj_all_e)
    #scores = [proj_all_e[rel*num_ent+headList[i]] +relListE[i] -  proj_all_e[rel*num_ent+tailList[i]]for i, rel in enumerate(relList)]
    #pickle.dump(scores, open('/scratch/symin95/TransINTOutputs/model/scores.p', 'wb'))
    #scores = torch.tensor([sum(subs**2) for subs in scores])
    
    #project the r's
    
    #shape is [num_ents x embed_dim] numpy array for each key 
    #pickle.dump(headList, open('/scratch/symin95/TransINTOutputs/model/headList.p', 'wb'))
    #scores = [sum(row**2) for row in headListE+relListE -tailListE]
    #MAYBE USE A NEW CUDA DEVICE HERE? 
    all_compares = torch.zeros(len(relList), num_ent, embedding_size)
    if head == 1 or head==0:
        for i, rel in enumerate(relList): 
            all_compares[i] = proj_all_e[rel*num_ent:(rel+1)*num_ent].cpu() + torch.cat([relListE[i].unsqueeze(0)]*num_ent).cpu()\
            -torch.cat([proj_all_e[rel*num_ent+tailList[i]].unsqueeze(0)]*num_ent).cpu() #- torch.cat([offListE[i].unsqueeze(0)]*num_ent)
            #pickle.dump(all_compares, open('/scratch/symin95/TransINTOutputs/model/all_compares.p', 'wb'))
        if embedding_size == 100:
            torch.cuda.set_device(3)
        all_compares = all_compares.cpu()
        if L1_flag == True:
            all_compares = torch.abs(all_compares)
            all_comp_scores = torch.sum(all_compares, dim=2) #size is [len(headList) x ent_num]
            all_compares = to_cuda(all_compares)
            all_comp_scores = to_cuda(all_comp_scores)
        else:
            all_compares = all_compares**2
            all_comp_scores = torch.sum(all_compares, dim=2)
            all_compares = to_cuda(all_compares)
            all_comp_scores = to_cuda(all_comp_scores)
            #
        score_sorted, arg_ranks = torch.sort(all_comp_scores, dim=1) #should be size #len(relList) x num_ent
        #print("arg ranks shape:", arg_ranks.shape)
        #print("arg ranks[0]:", arg_ranks[0])
            #print(arg_ranks)
            #print((arg_ranks == cur_head).nonzero())
            #print("sorted score:", score_sorted[(arg_ranks == cur_head).nonzero().item()])
            #print("all comp scores:", all_comp_scores[cur_head].item())
            
            #print("all compares head:", all_compares[cur_head])
            #print("pre-score i:", scores[i])
            #print((score_sorted==scores[i]).nonzero())
            ##just rank among all entities for one head, shape is [[1xnum_ents (total)]]
        if filter == False:
            rankListHead = (arg_ranks == to_cuda(torch.cat([torch.tensor([headList]).t()]*num_ent, dim=1))).nonzero()[:,1]
        else:
            rankListHead = argwhereHead(num_ent, headList, tailList, relList, arg_ranks, tripleDict, tripledict_given_tail_rel,impfilter = False)
        
        #print("rankListHead", rankListHead)
        rankListHead = rankListHead.type(torch.FloatTensor).cuda()
        isHit10ListHead = (rankListHead <10).type(torch.FloatTensor).cuda()
        isHit5ListHead = (rankListHead <5).type(torch.FloatTensor).cuda()
        isHit3ListHead = (rankListHead <3).type(torch.FloatTensor).cuda()
        isHit1ListHead = (rankListHead <1).type(torch.FloatTensor).cuda()
        #pickle.dump(isHit10ListHead, open("/scratch/symin95/isHit10ListHead.p","wb"))
        #pickle.dump(rankListHead, open("/scratch/symin95/rankListHead.p","wb"))
        #pickle.dump(len(testList), open("/scratch/symin95/len(testList).p","wb"))

        totalRank = sum(rankListHead)
        alltheRanks = rankListHead
        totalReciprocalRank = sum([1/float(elt+1) for elt in rankListHead])
        hit10Count = sum(isHit10ListHead); hit5Count = sum(isHit5ListHead);hit3Count = sum(isHit3ListHead);hit1Count = sum(isHit1ListHead)
        tripleCount = len(rankListHead)
    
    if head == 2 or head==0:
        for i, rel in enumerate(relList):
            all_compares[i] = torch.cat([proj_all_e[rel*num_ent+headList[i]].unsqueeze(0)]*num_ent) + torch.cat([relListE[i].unsqueeze(0)]*num_ent)\
            - proj_all_e[rel*num_ent:(rel+1)*num_ent] #- torch.cat([offListE[i].unsqueeze(0)]*num_ent)
        if embedding_size == 100:
            torch.cuda.set_device(3)
        all_compares = all_compares.cpu()
        if L1_flag == True:
            all_compares = torch.abs(all_compares)
            all_comp_scores = torch.sum(all_compares, dim=2) #size is [len(headList) x ent_num]
            all_compares = to_cuda(all_compares)
            all_comp_scores = to_cuda(all_comp_scores)
        else:
            all_compares = all_compares**2
            all_comp_scores = torch.sum(all_compares, dim=2)
            all_compares = to_cuda(all_compares)
            all_comp_scores = to_cuda(all_comp_scores)
            #
        score_sorted, arg_ranks = torch.sort(all_comp_scores, dim=1)  #This is just for the ith head now 
            ##just rank among all entities for one head, shape is [[1xnum_ents (total)]]
        if filter == False:
            rankListTail = (arg_ranks == to_cuda(torch.cat([torch.tensor([tailList]).t()]*num_ent, dim=1))).nonzero()[:,1]
        else:
            rankListTail = argwhereTail(num_ent, headList, tailList, relList, arg_ranks, tripleDict, tripledict_given_head_rel, impfilter=False)
        
        rankListTail = rankListTail.type(torch.FloatTensor).cuda()
        isHit10ListTail = (rankListTail <10).type(torch.FloatTensor).cuda()
        isHit5ListTail = (rankListTail <5).type(torch.FloatTensor).cuda()
        isHit3ListTail = (rankListTail <3).type(torch.FloatTensor).cuda()
        isHit1ListTail = (rankListTail <1).type(torch.FloatTensor).cuda()
        
        if head==2:
            totalRank = sum(rankListTail)
            alltheRanks = rankListTail
            totalReciprocalRank = sum([1/float(elt+1) for elt in rankListTail])
            hit10Count = sum(isHit10ListTail); hit5Count = sum(isHit5ListTail); hit3Count = sum(isHit3ListTail); hit1Count = sum(isHit1ListTail)
            tripleCount = len(rankListTail)
        elif head==0:
            totalRank += sum(rankListTail)
            alltheRanks = torch.cat([alltheRanks, rankListTail])
            totalReciprocalRank += sum([1/float(elt+1) for elt in rankListTail])
            hit10Count += sum(isHit10ListTail); hit5Count += sum(isHit5ListTail); hit3Count += sum(isHit3ListTail); hit1Count += sum(isHit1ListTail)
            tripleCount += len(rankListTail)

    #return hit10Count, hit5Count, hit3Count, totalRank, totalReciprocalRank, alltheRanks,  tripleCount
    if head == 1 or head == 2:
        hit10 = hit10Count / len(testList)
        hit5 = hit5Count / len(testList)
        hit3 = hit3Count / len(testList)
        hit1 = hit1Count / len(testList)
    
        meanrank = totalRank / len(testList)
        meanReciprocalRank = totalReciprocalRank / len(testList)
        medianRank = torch.median(alltheRanks)
        
    else:
        hit10 = hit10Count / len(2*testList)
        hit5 = hit5Count / len(2*testList)
        hit3 = hit3Count / len(2*testList)
        hit1 = hit1Count / len(2*testList)
    
        meanrank = totalRank / len(2*testList)
        meanReciprocalRank = totalReciprocalRank / len(2*testList)
        medianRank = torch.median(alltheRanks)

    print('Hit@10: %.6f' % hit10)
    print('Hit@5: %.6f' % hit5)
    print('Hit@3: %.6f' % hit3)
    print('Hit@1: %.6f' % hit1)
    print('Meanrank: %.6f' % meanrank)
    print('MeanReciprocalRank: %.6f' % meanReciprocalRank)
    print('MedianRank: %.6f' % medianRank)

    return hit10, hit5, hit3, meanrank, meanReciprocalRank, medianRank

class MyProcessTransINT(multiprocessing.Process):
    def __init__(self, L, tripleDict, ent_embeddings, 
        rel_embeddings, model, L1_flag, filter, queue=None, head=0, impfilter=False):
        super(MyProcessTransINT, self).__init__()
        self.L = L
        self.queue = queue
        self.tripleDict = tripleDict
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        self.L1_flag = L1_flag
        self.filter = filter
        self.head = head
        self.impfilter = impfilter
        self.model = model

    def run(self):
        while True:
            testList = self.queue.get()
            try:
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,self.model,
                     self.L1_flag, self.filter, self.L, self.head, self.impfilter)
            except:
                time.sleep(5)
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,self.model,
                     self.L1_flag, self.filter, self.L, self.head, self.impfilter)
            self.queue.task_done()

    def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
        model, L1_flag, filter, L, head, impfilter):

        hit10Count, hit5Count, hit3Count, totalRank, totalReciprocalRank, alltheRanks,  tripleCount = evaluation_transINT_helper(testList, tripleDict, ent_embeddings, 
            rel_embeddings, model,  L1_flag, filter, head, impfilter)

        L.append((hit10Count, hit5Count, hit3Count, totalRank, totalReciprocalRank, alltheRanks,  tripleCount))

def evaluation_transINT(testList, tripleDict, ent_embeddings, rel_embeddings, model,
    L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0, impfilter=False):
    # embeddings are torch tensor like (No Variable!)

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    # Split the testList according to the relation
    testList.sort(key=lambda x: (x.r, x.h, x.t))
    grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

    #ent_embeddings = ent_embeddings.cpu()
    #rel_embeddings = rel_embeddings.cpu()
    #norm_embeddings = norm_embeddings.cpu()

    with multiprocessing.Manager() as manager:
        L = manager.list()
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = MyProcessTransINT(L, tripleDict, ent_embeddings, rel_embeddings,model,
                 L1_flag, filter, queue=queue, head=head, impfilter=impfilter)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for k, subList in grouped:
            queue.put(subList)

        queue.join()

        resultList = list(L)

        for worker in workerList:
            worker.terminate()

    if head == 1 or head == 2:
        hit10 = sum([elem[0] for elem in resultList]) / len(testList)
        hit5 = sum([elem[1] for elem in resultList]) / len(testList)
        hit3 = sum([elem[2] for elem in resultList]) / len(testList)
    
        meanrank = sum([elem[3] for elem in resultList]) / len(testList)
        meanReciprocalRank = sum([elem[4] for elem in resultList]) / len(testList)
        medianRank = np.median(operator.reduce(operator.concat, list([elem[5] for elem in resultList])))
        
    else:
        hit10 = sum([elem[0] for elem in resultList]) / len(2*testList)
        hit5 = sum([elem[1] for elem in resultList]) / len(2*testList)
        hit3 = sum([elem[2] for elem in resultList]) / len(2*testList)
    
        meanrank = sum([elem[3] for elem in resultList]) / len(2*testList)
        meanReciprocalRank = sum([elem[4] for elem in resultList]) / len(2*testList)
        medianRank = np.median(operator.reduce(operator.concat, list([elem[5] for elem in resultList])))

    print('Hit@10: %.6f' % hit10)
    print('Hit@5: %.6f' % hit5)
    print('Hit@3: %.6f' % hit3)
    print('Meanrank: %.6f' % meanrank)
    print('MeanReciprocalRank: %.6f' % meanReciprocalRank)
    print('MedianRank: %.6f' % medianRank)

    return hit10, hit5, hit3, meanrank, meanReciprocalRank, medianRank

def evaluation_transH_helper(testList, tripleDict, ent_embeddings, 
    rel_embeddings, norm_embeddings, L1_flag, filter, head=0):
    # embeddings are torch tensor like (No Variable!)
    # Only one kind of relation

    headList = [triple.h for triple in testList]
    tailList = [triple.t for triple in testList]
    relList = [triple.r for triple in testList]

    h_e = ent_embeddings[headList]
    t_e = ent_embeddings[tailList]
    r_e = rel_embeddings[relList]
    this_rel = relList[0]
    this_norm_emb = norm_embeddings[this_rel]
    this_proj_all_e = projection_transH_pytorch(ent_embeddings, this_norm_emb)
    this_proj_all_e = this_proj_all_e.cpu().numpy()

    if head == 1:
        proj_t_e = projection_transH_pytorch(t_e, this_norm_emb)
        c_h_e = proj_t_e - r_e
        c_h_e = c_h_e.cpu().numpy()

        if L1_flag == True:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

        rankArrayHead = np.argsort(dist, axis=1)
        if filter == False:
            rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
        else:
            rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank = sum(rankListHead)
        hit10Count = len(isHit10ListHead)
        tripleCount = len(rankListHead)

    elif head == 2:
        proj_h_e = projection_transH_pytorch(h_e, this_norm_emb)
        c_t_e = proj_h_e + r_e
        c_t_e = c_t_e.cpu().numpy()

        if L1_flag == True:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

        rankArrayTail = np.argsort(dist, axis=1)
        if filter == False:
            rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
        else:
            rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayTail)]

        isHit10ListTail = [x for x in rankListTail if x < 10]

        totalRank = sum(rankListTail)
        hit10Count = len(isHit10ListTail)
        tripleCount = len(rankListTail)

    else:
        proj_h_e = projection_transH_pytorch(h_e, this_norm_emb)
        c_t_e = proj_h_e + r_e
        proj_t_e = projection_transH_pytorch(t_e, this_norm_emb)
        c_h_e = proj_t_e - r_e

        c_t_e = c_t_e.cpu().numpy()
        c_h_e = c_h_e.cpu().numpy()

        if L1_flag == True:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

        rankArrayTail = np.argsort(dist, axis=1)
        if filter == False:
            rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
        else:
            rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayTail)]

        isHit10ListTail = [x for x in rankListTail if x < 10]

        if L1_flag == True:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

        rankArrayHead = np.argsort(dist, axis=1)
        if filter == False:
            rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
        else:
            rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank = sum(rankListTail) + sum(rankListHead)
        hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
        tripleCount = len(rankListTail) + len(rankListHead)

    return hit10Count, totalRank, tripleCount

class MyProcessTransH(multiprocessing.Process):
    def __init__(self, L, tripleDict, ent_embeddings, 
        rel_embeddings, norm_embeddings, L1_flag, filter, queue=None, head=0):
        super(MyProcessTransH, self).__init__()
        self.L = L
        self.queue = queue
        self.tripleDict = tripleDict
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        self.norm_embeddings = norm_embeddings
        self.L1_flag = L1_flag
        self.filter = filter
        self.head = head

    def run(self):
        while True:
            testList = self.queue.get()
            try:
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
                    self.norm_embeddings, self.L1_flag, self.filter, self.L, self.head)
            except:
                time.sleep(5)
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
                    self.norm_embeddings, self.L1_flag, self.filter, self.L, self.head)
            self.queue.task_done()

    def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
        norm_embeddings, L1_flag, filter, L, head):

        hit10Count, totalRank, tripleCount = evaluation_transH_helper(testList, tripleDict, ent_embeddings, 
            rel_embeddings, norm_embeddings, L1_flag, filter, head)

        L.append((hit10Count, totalRank, tripleCount))

def evaluation_transH(testList, tripleDict, ent_embeddings, rel_embeddings, 
    norm_embeddings, L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0):
    # embeddings are torch tensor like (No Variable!)

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    # Split the testList according to the relation
    testList.sort(key=lambda x: (x.r, x.h, x.t))
    grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

    ent_embeddings = ent_embeddings.cpu()
    rel_embeddings = rel_embeddings.cpu()
    norm_embeddings = norm_embeddings.cpu()

    with multiprocessing.Manager() as manager:
        L = manager.list()
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = MyProcessTransH(L, tripleDict, ent_embeddings, rel_embeddings,
                norm_embeddings, L1_flag, filter, queue=queue, head=head)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for k, subList in grouped:
            queue.put(subList)

        queue.join()

        resultList = list(L)

        for worker in workerList:
            worker.terminate()

    if head == 1 or head == 2:
        hit10 = sum([elem[0] for elem in resultList]) / len(testList)
        meanrank = sum([elem[1] for elem in resultList]) / len(testList)
    else:
        hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
        meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

    print('Meanrank: %.6f' % meanrank)
    print('Hit@10: %.6f' % hit10)

    return hit10, meanrank

def evaluation_transR_helper(testList, tripleDict, ent_embeddings, 
    rel_embeddings, proj_embeddings, L1_flag, filter, head):
    # embeddings are torch tensor like (No Variable!)
    # Only one kind of relation

    headList = [triple.h for triple in testList]
    tailList = [triple.t for triple in testList]
    relList = [triple.r for triple in testList]

    h_e = ent_embeddings[headList]
    t_e = ent_embeddings[tailList]
    r_e = rel_embeddings[relList]
    this_rel = relList[0]
    this_proj_emb = proj_embeddings[[this_rel]]
    this_proj_all_e = projection_transR_pytorch(ent_embeddings, this_proj_emb)
    this_proj_all_e = this_proj_all_e.cpu().numpy()

    if head == 1:
        proj_t_e = projection_transR_pytorch(t_e, this_proj_emb)
        c_h_e = proj_t_e - r_e
        c_h_e = c_h_e.cpu().numpy()

        if L1_flag == True:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

        rankArrayHead = np.argsort(dist, axis=1)
        if filter == False:
            rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
        else:
            rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank = sum(rankListHead)
        hit10Count = len(isHit10ListHead)
        tripleCount = len(rankListHead)

    elif head == 2:
        proj_h_e = projection_transR_pytorch(h_e, this_proj_emb)
        c_t_e = proj_h_e + r_e
        c_t_e = c_t_e.cpu().numpy()

        if L1_flag == True:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

        rankArrayTail = np.argsort(dist, axis=1)
        if filter == False:
            rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
        else:
            rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayTail)]

        isHit10ListTail = [x for x in rankListTail if x < 10]

        totalRank = sum(rankListTail)
        hit10Count = len(isHit10ListTail)
        tripleCount = len(rankListTail)

    else:
        proj_h_e = projection_transR_pytorch(h_e, this_proj_emb)
        c_t_e = proj_h_e + r_e
        proj_t_e = projection_transR_pytorch(t_e, this_proj_emb)
        c_h_e = proj_t_e - r_e

        c_t_e = c_t_e.cpu().numpy()
        c_h_e = c_h_e.cpu().numpy()

        if L1_flag == True:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

        rankArrayTail = np.argsort(dist, axis=1)
        if filter == False:
            rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
        else:
            rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayTail)]

        isHit10ListTail = [x for x in rankListTail if x < 10]

        if L1_flag == True:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

        rankArrayHead = np.argsort(dist, axis=1)
        if filter == False:
            rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
        else:
            rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank = sum(rankListTail) + sum(rankListHead)
        hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
        tripleCount = len(rankListTail) + len(rankListHead)

    return hit10Count, totalRank, tripleCount

class MyProcessTransR(multiprocessing.Process):
    def __init__(self, L, tripleDict, ent_embeddings, 
        rel_embeddings, proj_embeddings, L1_flag, filter, queue=None, head=0):
        super(MyProcessTransR, self).__init__()
        self.L = L
        self.queue = queue
        self.tripleDict = tripleDict
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        self.proj_embeddings = proj_embeddings
        self.L1_flag = L1_flag
        self.filter = filter
        self.head = head

    def run(self):
        while True:
            testList = self.queue.get()
            try:
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
                    self.proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
            except:
                time.sleep(5)
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
                    self.proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
            self.queue.task_done()

    def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
        proj_embeddings, L1_flag, filter, L, head):

        hit10Count, totalRank, tripleCount = evaluation_transR_helper(testList, tripleDict, ent_embeddings, 
            rel_embeddings, proj_embeddings, L1_flag, filter, head=head)

        L.append((hit10Count, totalRank, tripleCount))

def evaluation_transR(testList, tripleDict, ent_embeddings, rel_embeddings, 
    proj_embeddings, L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0):
    # embeddings are torch tensor like (No Variable!)

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    # Split the testList according to the relation
    testList.sort(key=lambda x: (x.r, x.h, x.t))
    grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

    ent_embeddings = ent_embeddings.cpu()
    rel_embeddings = rel_embeddings.cpu()
    proj_embeddings = proj_embeddings.cpu()

    with multiprocessing.Manager() as manager:
        L = manager.list()
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = MyProcessTransR(L, tripleDict, ent_embeddings, rel_embeddings,
                proj_embeddings, L1_flag, filter, queue=queue, head=head)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for k, subList in grouped:
            queue.put(subList)

        queue.join()

        resultList = list(L)

        for worker in workerList:
            worker.terminate()

    if head == 1 or head == 2:
        hit10 = sum([elem[0] for elem in resultList]) / len(testList)
        meanrank = sum([elem[1] for elem in resultList]) / len(testList)    
    else:
        hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
        meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

    print('Meanrank: %.6f' % meanrank)
    print('Hit@10: %.6f' % hit10)

    return hit10, meanrank

def evaluation_transD_helper(testList, tripleDict, ent_embeddings, 
    rel_embeddings, ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, head=0):
    # embeddings are torch tensor like (No Variable!)
    # Only one kind of relation

    headList = [triple.h for triple in testList]
    tailList = [triple.t for triple in testList]
    relList = [triple.r for triple in testList]

    h_e = ent_embeddings[headList]
    t_e = ent_embeddings[tailList]
    r_e = rel_embeddings[relList]
    head_proj_emb = ent_proj_embeddings[headList]
    tail_proj_emb = ent_proj_embeddings[tailList]
    this_rel = relList[0]
    this_rel_proj_emb = rel_proj_embeddings[[this_rel]]
    this_proj_all_e = projection_transD_pytorch_samesize(ent_embeddings, ent_proj_embeddings, this_rel_proj_emb)
    this_proj_all_e = this_proj_all_e.cpu().numpy()

    if head == 1:
        proj_t_e = projection_transD_pytorch_samesize(t_e, tail_proj_emb, this_rel_proj_emb)
        c_h_e = proj_t_e - r_e
        c_h_e = c_h_e.cpu().numpy()

        if L1_flag == True:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

        rankArrayHead = np.argsort(dist, axis=1)
        if filter == False:
            rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
        else:
            rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank = sum(rankListHead)
        hit10Count = len(isHit10ListHead)
        tripleCount = len(rankListHead)

    elif head == 2:
        proj_h_e = projection_transD_pytorch_samesize(h_e, head_proj_emb, this_rel_proj_emb)
        c_t_e = proj_h_e + r_e
        c_t_e = c_t_e.cpu().numpy()

        if L1_flag == True:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

        rankArrayTail = np.argsort(dist, axis=1)
        if filter == False:
            rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
        else:
            rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayTail)]

        isHit10ListTail = [x for x in rankListTail if x < 10]

        totalRank = sum(rankListTail)
        hit10Count = len(isHit10ListTail)
        tripleCount = len(rankListTail)

    else:
        proj_h_e = projection_transD_pytorch_samesize(h_e, head_proj_emb, this_rel_proj_emb)
        c_t_e = proj_h_e + r_e
        proj_t_e = projection_transD_pytorch_samesize(t_e, tail_proj_emb, this_rel_proj_emb)
        c_h_e = proj_t_e - r_e

        c_t_e = c_t_e.cpu().numpy()
        c_h_e = c_h_e.cpu().numpy()

        if L1_flag == True:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

        rankArrayTail = np.argsort(dist, axis=1)
        if filter == False:
            rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
        else:
            rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayTail)]

        isHit10ListTail = [x for x in rankListTail if x < 10]

        if L1_flag == True:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

        rankArrayHead = np.argsort(dist, axis=1)
        if filter == False:
            rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
        else:
            rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
                            for elem in zip(headList, tailList, relList, rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank = sum(rankListTail) + sum(rankListHead)
        hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
        tripleCount = len(rankListTail) + len(rankListHead)

    return hit10Count, totalRank, tripleCount

class MyProcessTransD(multiprocessing.Process):
    def __init__(self, L, tripleDict, ent_embeddings, 
        rel_embeddings, ent_proj_embeddings, rel_proj_embeddings,L1_flag, filter, queue=None, head=0):
        super(MyProcessTransD, self).__init__()
        self.L = L
        self.queue = queue
        self.tripleDict = tripleDict
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        self.ent_proj_embeddings = ent_proj_embeddings
        self.rel_proj_embeddings = rel_proj_embeddings
        self.L1_flag = L1_flag
        self.filter = filter
        self.head = head

    def run(self):
        while True:
            testList = self.queue.get()
            try:
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
                    self.ent_proj_embeddings, self.rel_proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
            except:
                time.sleep(5)
                self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
                    self.ent_proj_embeddings, self.rel_proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
            self.queue.task_done()

    def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
        ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, L, head):

        hit10Count, totalRank, tripleCount = evaluation_transD_helper(testList, tripleDict, ent_embeddings, 
            rel_embeddings, ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, head=head)

        L.append((hit10Count, totalRank, tripleCount))

def evaluation_transD(testList, tripleDict, ent_embeddings, rel_embeddings, 
    ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0):
    # embeddings are torch tensor like (No Variable!)

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    # Split the testList according to the relation
    testList.sort(key=lambda x: (x.r, x.h, x.t))
    grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

    ent_embeddings = ent_embeddings.cpu()
    rel_embeddings = rel_embeddings.cpu()
    ent_proj_embeddings = ent_proj_embeddings.cpu()
    rel_proj_embeddings = rel_proj_embeddings.cpu()

    with multiprocessing.Manager() as manager:
        L = manager.list()
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = MyProcessTransD(L, tripleDict, ent_embeddings, rel_embeddings,
                ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, queue=queue, head=head)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for k, subList in grouped:
            queue.put(subList)

        queue.join()

        resultList = list(L)

        for worker in workerList:
            worker.terminate()

    if head == 1 or head == 2:
        hit10 = sum([elem[0] for elem in resultList]) / len(testList)
        meanrank = sum([elem[1] for elem in resultList]) / len(testList)
    else:
        hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
        meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

    print('Meanrank: %.6f' % meanrank)
    print('Hit@10: %.6f' % hit10)

    return hit10, meanrank
