#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-24 01:45:03
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import datetime
import random

from utils import *
from data import *
from evaluation import *
import loss
import model
import pickle
import time

#from hyperboard import Agent

USE_CUDA = torch.cuda.is_available()
print("using GPU? ", USE_CUDA)

if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor

"""
The meaning of parameters:
self.dataset: Which dataset is used to train the model? Such as 'FB15k', 'WN18', etc.
self.learning_rate: Initial learning rate (lr) of the model.
self.early_stopping_round: How many times will lr decrease? If set to 0, it remains constant.
self.L1_flag: If set to True, use L1 distance as dissimilarity; else, use L2.
self.embedding_size: The embedding size of entities and relations.
self.num_batches: How many batches to train in one epoch?
self.train_times: The maximum number of epochs for training.
self.margin: The margin set for MarginLoss.
self.filter: Whether to check a generated negative sample is false negative.
self.momentum: The momentum of the optimizer.
self.optimizer: Which optimizer to use? Such as SGD, Adam, etc.
self.loss_function: Which loss function to use? Typically, we use margin loss.
self.entity_total: The number of different entities.
self.relation_total: The number of different relations.
self.batch_size: How many instances is contained in one batch?
"""

class Config(object):
    def __init__(self):
        self.dataset = None
        self.learning_rate = 0.001
        self.early_stopping_round = 0
        self.L1_flag = False
        self.embedding_size = 100
        self.num_batches = 100
        self.train_times = 1000
        self.margin = 1.0
        self.filter = True
        self.momentum = 0.9
        self.optimizer = optim.Adam
        self.loss_function = loss.marginLoss
        self.entity_total = 0
        self.relation_total = 0
        self.batch_size = 0
        self.impfilter = False
        self.which_dataset = None

if __name__ == "__main__":

    import argparse
    argparser = argparse.ArgumentParser()

    """
    The meaning of some parameters:
    seed: Fix the random seed. Except for 0, which means no setting of random seed.
    port: The port number used by hyperboard, 
    which is a demo showing training curves in real time.
    You can refer to https://github.com/WarBean/hyperboard to know more.
    num_processes: Number of processes used to evaluate the result.
    """

    argparser.add_argument('-d', '--dataset', type=str)
    argparser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    argparser.add_argument('-es', '--early_stopping_round', type=int, default=0)
    argparser.add_argument('-L', '--L1_flag', type=int, default=2)
    argparser.add_argument('-em', '--embedding_size', type=int, default=100)
    argparser.add_argument('-nb', '--num_batches', type=int, default=100)
    argparser.add_argument('-n', '--train_times', type=int, default=1000)
    argparser.add_argument('-m', '--margin', type=float, default=1.0)
    argparser.add_argument('-f', '--filter', type=int, default=1)
    argparser.add_argument('-mo', '--momentum', type=float, default=0.9)
    argparser.add_argument('-s', '--seed', type=int, default=0)
    argparser.add_argument('-op', '--optimizer', type=int, default=1)
    argparser.add_argument('-lo', '--loss_type', type=int, default=0)
    argparser.add_argument('-p', '--port', type=int, default=5000)
    argparser.add_argument('-np', '--num_processes', type=int, default=4)
    argparser.add_argument('-norm_reg_rel', '--norm_reg_rel', type=int, default=0)
    argparser.add_argument('-lr_decay', '--lr_decay', type=float, default=0.5)
    argparser.add_argument('-c', '--cuda', type=int, default=0)
    argparser.add_argument('-rank_file', '--rank_file', type=str, required=True) 
    argparser.add_argument('-imp', '--impfilter', type=int, required=True) 
    argparser.add_argument('-gray', '--gray', type=int, required=True) 
    argparser.add_argument('-train_test', '--train_test', type=int, default=0) 
    argparser.add_argument('-transH_test', '--transH_test', type=int, default=0) 
    argparser.add_argument('-norm_rel', '--norm_rel', type=int, default=0) 
    argparser.add_argument('-norm_ent', '--norm_ent', type=int, default=0) 
    argparser.add_argument('-norm_mat', '--norm_mat', type=int, default=0) 
    argparser.add_argument('-pbt', '--print_batch_time', type=int, default=0) 
    argparser.add_argument('-jte', '--just_try_eval', type=int, default=0) 
    argparser.add_argument('-eval_cycle', '--eval_cycle', type=int, default=10) 
    argparser.add_argument('-which_dataset', '--which_dataset', type=str, default='wn18') 


    args = argparser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(args.cuda)
    else:
        device = torch.device('cpu')

    # Start the hyperboard agent
    #agent = Agent(address='127.0.0.1', port=args.port)

    if args.seed != 0:
        torch.manual_seed(args.seed)
        
    if args.which_dataset == 'wn18':
        dataset_folder = 'datasets/KALE_datasets/wn18/'
    elif args.which_dataset == 'fb122':
        dataset_folder = 'datasets/KALE_datasets/fb122/'
    elif args.which_dataset == 'Nell_loc':
        dataset_folder = 'datasets/NELL_datasets/location/'
    elif args.which_dataset == 'Nell_sp':
        dataset_folder = 'datasets/NELL_datasets/sports/'

    #trainTotal, trainList, trainDict = loadTriple('./datasets/' + args.dataset, 'train2id.txt')
    #validTotal, validList, validDict = loadTriple('./datasets/' + args.dataset, 'valid2id.txt')
    #tripleTotal, tripleList, tripleDict = loadTriple('./datasets/' + args.dataset, 'triple2id.txt')
    trainList = [Triple(tr[0], tr[1], tr[2]) for tr in pickle.load(open(dataset_folder + 'train_triples_list.p', 'rb'))]
    trainTotal = len(trainList)
    testList = [Triple(tr[0], tr[1], tr[2]) for tr in pickle.load(open(dataset_folder + 'test_triples_list.p', 'rb'))]
    testTotal = len(testList)
    try:
        validList = [Triple(tr[0], tr[1], tr[2]) for tr in pickle.load(open(dataset_folder + 'tval_triples_list.p', 'rb'))]
        validTotal = len(validList)
    except:
        validList = testList
        validTotal = testTotal
    tripleDict = pickle.load( open(dataset_folder + 'tripleDict.p','rb'))
    tail_per_head = pickle.load( open(dataset_folder +'tail_per_head.p', 'rb'))
    head_per_tail = pickle.load(open(dataset_folder +'head_per_tail.p', 'rb'))
    possible_imp_given_tail_rel= pickle.load(open(dataset_folder +'possible_imp_given_tail_rel','rb'))
    possible_imp_given_head_rel =  pickle.load(open(dataset_folder +'possible_imp_given_head_rel','rb'))
    tripledict_given_tail_rel = pickle.load(open(dataset_folder +'tripledict_given_tail_rel','rb'))
    tripledict_given_head_rel = pickle.load(open(dataset_folder +'tripledict_given_head_rel','rb'))
        
    config = Config()
    config.dataset = args.dataset
    config.learning_rate = args.learning_rate
    config.which_dataset = args.which_dataset


    config.early_stopping_round = args.early_stopping_round

    if args.L1_flag == 1:
        config.L1_flag = True
    else:
        config.L1_flag = False
    if args.impfilter == 1:
        config.impfilter = True

    config.embedding_size = args.embedding_size
    config.num_batches = args.num_batches
    config.train_times = args.train_times
    config.margin = args.margin

    if args.filter == 1:
        config.filter = True
    else:
        config.filter = False

    config.momentum = args.momentum

    if args.optimizer == 0:
        config.optimizer = optim.SGD
    elif args.optimizer == 1:
        config.optimizer = optim.Adam
    elif args.optimizer == 2:
        config.optimizer = optim.RMSprop

    if args.loss_type == 0:
        config.loss_function = loss.marginLoss

    config.entity_total = len(pickle.load(open(dataset_folder +'ent_idx.p', 'rb'))['id2ent_dict'])
    config.relation_total = len(pickle.load(open(dataset_folder +'rel_idx.p', 'rb'))['id2rel_dict'])
    config.batch_size = trainTotal // config.num_batches
    
    #Load data from the rules 
    rank_file = pickle.load(open(dataset_folder + args.rank_file , 'rb'))
    
    r_id2_vrr = pickle.load(open(dataset_folder + 'r_id2_vrr.p', 'rb')) 
    try:
        equiv_flip_pairs = pickle.load(open(dataset_folder + 'equiv_flip_pairs.p', 'rb'))  
        equiv_flip_dict = pickle.load(open(dataset_folder + 'equiv_flip_dict.p', 'rb'))  
    except:
        equiv_flip_pairs = {}
        equiv_flip_dict = {}
    #in_bundle = pickle.load(open('datasets/KALE_datasets/fb122/in_bundle.p', 'rb')) 
    #equiv_child_dict = pickle.load(open('datasets/KALE_datasets/fb122/equiv_child_dict.p','rb'))

    loss_function = config.loss_function()
    if args.transH_test ==0:
        model = model.TransINTModel(config, rank_file, r_id2_vrr, equiv_flip_dict)
    else:
        model = model.TransINTModelTestTransH(config, rank_file, r_id2_vrr, equiv_flip_dict)

    if USE_CUDA:
        model.cuda()
        loss_function.cuda()

    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    margin = autograd.Variable(floatTensor([config.margin]))

    start_time = time.time()

    foldername = '_'.join(
        ['l', str(args.learning_rate),
         'es', str(args.early_stopping_round),
         'L', str(args.L1_flag),
         'em', str(args.embedding_size),
         'nb', str(args.num_batches),
         'n', str(args.train_times),
         'm', str(args.margin),
         'f', str(args.filter),
         'mo', str(args.momentum),
         's', str(args.seed),
         'op', str(args.optimizer),
         'lo', str(args.loss_type),]) 
    filename = foldername + '/TransINT'

    trainBatchList = getBatchList(trainList, config.num_batches)
    #print("in evaluation: ", [(trip.h, trip.t, trip.r) for trip in trainList[:config.batch_size]])
    
    if args.just_try_eval == 1:
        eval_time = time.time()
        ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy() #weight of all the emebeddings
        if args.transH_test == 1:
            rel_embeddings = model.vvrel_embeddings.weight.data.cpu().numpy()
        else:
            rel_embeddings = model.vvrel_all_weights()
        L1_flag = model.L1_flag
        filter = model.filter
        hit10, hit5, hit3, now_meanrank, meanReciprocalRank, medianRank = evaluation_transINT_helper(trainList[:916], tripleDict, ent_embeddings, rel_embeddings,
                        model, L1_flag, filter, tripledict_given_tail_rel, tripledict_given_head_rel, 0,  0, impfilter=config.impfilter)
        torch.cuda.set_device(args.cuda)

        print("eval time:",time.time() - eval_time)
    for epoch in range(config.train_times):
    ##for epoch in range(1):
        total_loss = floatTensor([0.0])
        ###############
        ############### TAKE CARE OF THIS!
        #############
        if args.train_test ==0:
            random.shuffle(trainBatchList)
        ###############
        ############### TAKE CARE OF THIS!
        #############
        st = time.time()
        for batchList in trainBatchList: #batchList is list of Triples 
            #batch_time = time.time()
            if config.impfilter == True:
                pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_filter_all_v2(batchList, 
                    config.entity_total, tail_per_head, head_per_tail, possible_imp_given_tail_rel, possible_imp_given_head_rel, tripleDict, True) #config.entity_total is total number of entities 
                #list of correct triple's h,r,t and corrupt triple's h,r,t
                #dim is [batch_size] for all pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch
            if config.impfilter == False:
                pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_filter_all_v2(batchList, 
                    config.entity_total, tail_per_head, head_per_tail, possible_imp_given_tail_rel, possible_imp_given_head_rel, tripleDict, False)
            #print("batch time:", batch_time-time.time())
# =============================================================================
#             batch_entity_set = set(pos_h_batch + pos_t_batch + neg_h_batch + neg_t_batch)
#             batch_relation_set = set(pos_r_batch + neg_r_batch)
#             batch_entity_list = list(batch_entity_set)
#             batch_relation_list = list(batch_relation_set)
# =============================================================================
            long_ten_time = time.time()
            pos_h_batch = autograd.Variable(longTensor(pos_h_batch)) #[batch_size]
            pos_t_batch = autograd.Variable(longTensor(pos_t_batch)) #[batch_size]
            pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
            neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
            neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
            neg_r_batch = autograd.Variable(longTensor(neg_r_batch))
            #print("long tensor time:", long_ten_time-time.time())
            
            zero_time = time.time()
            model.zero_grad()
            #print("zero grad time:", time.time()-zero_time)
            mod_time = time.time() 
            pos, neg = model(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch) 
            #print("mod time :", time.time()-mod_time)
            #pos, neg are each [distance * batch_size] for list (len == batch_size) of correct and corrupt triples
            #if epoch %10 ==0:
            #    print("pos-neg:", pos-neg)
            if args.loss_type == 0:
                losses = loss_function(pos, neg, margin) #margin is applied to every (pos_distance, neg_distance) and summed
            else:
                losses = loss_function(pos, neg)
                        #check shape 
            #print("ENT EMBEDDING SHAPE: " , ent_embeddings.shape)
            #print("REL EMBEDDING SHAPE: " , rel_embeddings.shape)
            #losses = losses #+ loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings) #Are these regularization? #What are these? 
            #losses += loss.normLoss(ent_embeddings) 
            if args.norm_reg_rel == 1:
                ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch])) 
                #rel_embeddings = model.vvrel_embedding_func(torch.cat([pos_r_batch]))
                losses+= loss.normLoss(ent_embeddings) #+ loss.normLoss(ent_embeddings) 
            backward_tim = time.time()
            losses.backward()
            optimizer.step()
            total_loss += losses.data
            #print("backward time :", time.time()-backward_tim)
            
            #Normalize the rand vecs and put them back
        if args.transH_test ==0:
            if args.norm_mat == 1:
                model.normalize_all_rows()
            if args.norm_ent==1:
                model.normalize_ent_embeddings()                   
            if args.norm_rel==1:
                model.normalize_vvrel_embeddings()
        else:
            normalize_norm_emb = F.normalize(model.bases.data[longTensor(batch_relation_list)], p=2, dim=1)
            model.bases.data[longTensor(batch_relation_list)] = normalize_norm_emb    
            if args.norm_ent==1:
                normalize_ent_emb = F.normalize(model.ent_embeddings.weight.data[longTensor(batch_relation_list)], p=2, dim=1)
                model.ent_embeddings.weight.data[longTensor(batch_relation_list)] = normalize_ent_emb    
                
            if args.norm_rel==1:
                normalize_rel_emb = F.normalize(model.vvrel_embeddings.weight.data[longTensor(batch_relation_list)], p=2, dim=1)
                model.vvrel_embeddings.weight.data[longTensor(batch_relation_list)] = normalize_rel_emb   
            
            #pickle.dump(model.vvr_bundle_heads, open('/scratch/symin95/TransINTOutputs/tempcheck/ep:'+str(epoch) +'vvr_bundle_heads.p', 'wb'))
            #pickle.dump(model.dict_of_random_mats, open('/scratch/symin95/TransINTOutputs/tempcheck/ep:'+str(epoch) +'dict_of_random_mats.p', 'wb'))
            #pickle.dump(model.dict_of_random_mats, open('/scratch/symin95/TransINTOutputs/tempcheck/ep:'+str(epoch) +'dict_of_random_mats.p', 'wb'))
            
            if args.print_batch_time == 1:
                print("time took per batch :", st - time.time())
            if args.train_test==1:
                break
        #agent.append(trainCurve, epoch, total_loss[0])

        #Print time elapsed and loss
        if epoch % 10 == 0:
            now_time = time.time()
            print("===========================================")
            print("Epoch ", epoch)
            print("time elapsed: ", now_time - start_time)
            print("Train Total loss: %f" % (total_loss[0]))
        
        #Validation, just loss
        #with no grad 
# =============================================================================
#         if epoch % 10 == 0:
#             if config.filter == True:
#                 pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_filter_random_v2(validList, 
#                     config.batch_size, config.entity_total, tripleDict,tail_per_head, head_per_tail,equiv_child_dict, config.impfilter)
#             else:
#                 pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_raw_random_v2(validList, 
#                     config.batch_size, config.entity_total, tail_per_head, head_per_tail)
#             pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
#             pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
#             pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
#             neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
#             neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
#             neg_r_batch = autograd.Variable(longTensor(neg_r_batch))
# 
#             pos, neg = model(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch)
# 
#             if args.loss_type == 0:
#                 losses = loss_function(pos, neg, margin)
#             else:
#                 losses = loss_function(pos, neg)
#             ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
#             rel_embeddings = model.vvrel_embedding_func(torch.cat([pos_r_batch, neg_r_batch]))
#             losses = losses #+ loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings)
#             #losses += loss.normLoss(ent_embeddings)
#             if args.norm_reg_rel == 1:
#                  losses += loss.normLoss(rel_embeddings)+ loss.normLoss(ent_embeddings) 
#             #print("losses: ", losses)
#             print("Validation batch loss: %f" % (losses.data))
#             #agent.append(validCurve, epoch, losses.data[0])
# =============================================================================

        #Actual Evaluation Validation, decrease of lr
        if config.early_stopping_round > 0:
            if epoch == 0:
# =============================================================================
#                 ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
#                 rel_embeddings = model.vvrel_all_weights()
#                 L1_flag = model.L1_flag
#                 filter = model.filter    
#                 print("--------- validation for epoch" + str(epoch) + " ----------------------------")
#                 #hit10, hit5, hit3, meanrank, meanReciprocalRank, medianRank = evaluation_transINT(validList, tripleDict, ent_embeddings, rel_embeddings, 
#                 #    model, L1_flag, filter, 10 * config.batch_size, num_processes=args.num_processes, impfilter=config.impfilter)
#                 hit10, hit5, hit3, best_meanrank, meanReciprocalRank, medianRank = evaluation_transINT_helper(validList, tripleDict, ent_embeddings, rel_embeddings, 
#                     model, L1_flag, filter, equiv_child_dict,  config.batch_size, impfilter=config.impfilter)
# =============================================================================
                best_medianrank = 10000000
                #agent.append(hit10Curve, epoch, hit10)
                #agent.append(meanrankCurve, epoch, best_meanrank)
                if args.gray ==1:
                    if not os.path.exists(os.path.join('/scratch/symin95/TransINTOutputs/model/' + args.which_dataset + '/' + args.dataset, foldername)):
                        os.makedirs(os.path.join('/scratch/symin95/TransINTOutputs/model/' +  args.which_dataset + '/' +  args.dataset, foldername))
                    torch.save(model, os.path.join('/scratch/symin95/TransINTOutputs/model/' + args.which_dataset + '/' +  args.dataset, filename + "_epoch_" + str(epoch) + ".ckpt" ))  
                else:
                    if not os.path.exists(os.path.join('./model/' +  args.which_dataset + '/' + args.dataset, foldername)):
                        os.makedirs(os.path.join('./model/' +  args.which_dataset + '/' + args.dataset, foldername))
                    torch.save(model, os.path.join('./model/' +  args.which_dataset + '/' + args.dataset, filename + "_epoch_" + str(epoch) + ".ckpt"))    
                best_epoch = 0
                meanrank_not_decrease_time = 0
                lr_decrease_time = 0
                #if USE_CUDA:
                    #model.cuda()

            # Evaluate on validation set for every 5 epochs
            elif epoch % args.eval_cycle == 0:
                ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy() #weight of all the emebeddings
                if args.transH_test == 1:
                    rel_embeddings = model.vvrel_embeddings.weight.data.cpu().numpy()
                else:
                    rel_embeddings = model.vvrel_all_weights()
                L1_flag = model.L1_flag
                filter = model.filter
                print("--------- validation for epoch" + str(epoch) + " ----------------------------")
                #hit10, hit5, hit3, meanrank, meanReciprocalRank, medianRank = evaluation_transINT(validList, tripleDict, ent_embeddings, rel_embeddings, 
                #    model, L1_flag, filter, 10* config.batch_size, num_processes=args.num_processes, impfilter=config.impfilter)
                if args.train_test ==1:
                    hit10, hit5, hit3, meanrank, meanReciprocalRank, now_medianRank = evaluation_transINT_helper(trainList[:config.batch_size], tripleDict, ent_embeddings, rel_embeddings, 
                            model, L1_flag, filter, tripledict_given_tail_rel, tripledict_given_head_rel, config.batch_size, 0,  impfilter=config.impfilter)
                else:
                    hit10, hit5, hit3, meanrank, meanReciprocalRank, now_medianRank = evaluation_transINT_helper(validList, tripleDict, ent_embeddings, rel_embeddings,
                        model, L1_flag, filter, tripledict_given_tail_rel, tripledict_given_head_rel, 900,  0, impfilter=config.impfilter)
                torch.cuda.set_device(args.cuda)
                #agent.append(hit10Curve, epoch, hit10)
                #agent.append(meanrankCurve, epoch, now_meanrank)
                if now_medianRank <= best_medianrank:
                    meanrank_not_decrease_time = 0
                    best_medianrank = now_medianRank
                    best_epoch = epoch
                    if args.gray ==1:
                        if not os.path.exists(os.path.join('/scratch/symin95/TransINTOutputs/model/' + args.which_dataset + '/'+ args.dataset, foldername)):
                            os.makedirs(os.path.join('/scratch/symin95/TransINTOutputs/model/' +   args.which_dataset + '/'+ args.dataset, foldername))
                        torch.save(model, os.path.join('/scratch/symin95/TransINTOutputs/model/' +  args.which_dataset + '/'+ args.dataset, filename + "_epoch_" + str(epoch) + ".ckpt"))  
                        file = open(os.path.join('/scratch/symin95/TransINTOutputs/model/' +  args.which_dataset + '/'+ args.dataset, foldername, "log.txt"),"w")
                    else:
                        if not os.path.exists(os.path.join('./model/' + args.which_dataset + '/'+ args.dataset, foldername)):
                            os.makedirs(os.path.join('./model/' +  args.which_dataset + '/'+args.dataset, foldername))
                        torch.save(model, os.path.join('./model/' +  args.which_dataset + '/'+ args.dataset, filename + "_epoch_" + str(epoch) + ".ckpt"))    
                        file = open(os.path.join('./model/' +  args.which_dataset + '/'+ args.dataset, foldername, "log.txt"),"w")
                    file.write("best mean rank epoch at ckpt: " + str(best_epoch) +"\n")
                    file.write("best mean rank at ckpt: " + str(best_medianrank) +"\n")
                    file.close()

 
                else:
                    meanrank_not_decrease_time += 1
                    # If the result hasn't improved for consecutive 5 evaluations, decrease learning rate
                    if meanrank_not_decrease_time == 5:
                        lr_decrease_time += 1
                        if lr_decrease_time == config.early_stopping_round:
                            break
                        else:
                            #optimizer.param_groups[0]['lr'] *= args.lr_decay #0.5
                            meanrank_not_decrease_time = 0
                print("best validation mean rank so far: ", best_medianrank, " at epoch: ", best_epoch)
                optimizer.param_groups[0]['lr'] *= args.lr_decay

                #if USE_CUDA:
                    #model.cuda()
        
        #No early stopping
        elif (epoch + 1) % 10 == 0 or epoch == 0:
            if args.gray ==1:
                if not os.path.exists(os.path.join('/scratch/symin95/TransINTOutputs/model/' +  args.which_dataset + '/'+args.dataset, foldername)):
                    os.makedirs(os.path.join('/scratch/symin95/TransINTOutputs/model/' +  args.which_dataset + '/'+args.dataset, foldername))
                torch.save(model, os.path.join('/scratch/symin95/TransINTOutputs/model/' + args.which_dataset + '/'+ args.dataset, filename + "_epoch_" + str(epoch) + ".ckpt"))  
            else:
                if not os.path.exists(os.path.join('./model/' +  args.which_dataset + '/'+args.dataset, foldername)):
                    os.makedirs(os.path.join('./model/' +  args.which_dataset + '/'+args.dataset, foldername))
                torch.save(model, os.path.join('./model/' +  args.which_dataset + '/'+args.dataset, filename + "_epoch_" + str(epoch) + ".ckpt"))    

    #testTotal, testList, testDict = loadTriple('./datasets/' + args.dataset, 'test2id.txt')
# =============================================================================
#     oneToOneTotal, oneToOneList, oneToOneDict = loadTriple('./datasets/' + args.dataset, 'one_to_one_test.txt')
#     oneToManyTotal, oneToManyList, oneToManyDict = loadTriple('./datasets/' + args.dataset, 'one_to_many_test.txt')
#     manyToOneTotal, manyToOneList, manyToOneDict = loadTriple('./datasets/' + args.dataset, 'many_to_one_test.txt')
#     manyToManyTotal, manyToManyList, manyToManyDict = loadTriple('./datasets/' + args.dataset, 'many_to_many_test.txt')
# 
# =============================================================================
    ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
    rel_embeddings = model.vvrel_all_weights()
    L1_flag = model.L1_flag
    filter = model.filter
    
    print("--------- all of test ----------------------------")
    print("run again")
    #hit10, hit5, hit3, meanrank, meanReciprocalRank, medianRank = evaluation_transINT(testList, tripleDict, ent_embeddings, rel_embeddings, model, L1_flag, filter, head=0, impfilter=config.impfilter)
    #hit10, hit5, hit3, meanrank, meanReciprocalRank, medianRank = evaluation_transINT_helper(testList, tripleDict, ent_embeddings, rel_embeddings, model, L1_flag, filter,0, head=0, impfilter=config.impfilter)

# =============================================================================
#     print("--------- 1-to-1 HEAD test ----------------------------")
#     hit10OneToOneHead, meanrankOneToOneHead = evaluation_transE(oneToOneList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=1)
#     print("--------- 1-to-Many HEAD test ----------------------------")
#     hit10OneToManyHead, meanrankOneToManyHead = evaluation_transE(oneToManyList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=1)
#     print("--------- Many-to-1 HEAD test ----------------------------")
#     hit10ManyToOneHead, meanrankManyToOneHead = evaluation_transE(manyToOneList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=1)
#     print("--------- Many-to-Many HEAD test ----------------------------")
#     hit10ManyToManyHead, meanrankManyToManyHead = evaluation_transE(manyToManyList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=1)
# 
#     print("--------- 1-to-1 TAIL test ----------------------------")
#     hit10OneToOneTail, meanrankOneToOneTail = evaluation_transE(oneToOneList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=2)
#     print("--------- 1-to-Many TAIL test ----------------------------")
#     hit10OneToManyTail, meanrankOneToManyTail = evaluation_transE(oneToManyList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=2)
#     print("--------- Many-to-1 TAIL test ----------------------------")
#     hit10ManyToOneTail, meanrankManyToOneTail = evaluation_transE(manyToOneList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=2)
#     print("--------- Many-to-Many TAIL test ----------------------------")
#     hit10ManyToManyTail, meanrankManyToManyTail = evaluation_transE(manyToManyList, tripleDict, ent_embeddings, rel_embeddings, L1_flag, filter, head=2)    
# 
# =============================================================================
# =============================================================================
#     writeList = [filename, 
#         'testSet', '%.6f' % hit10, '%.6f' % hit5, '%.6f' % hit3,  
#         '%.6f' % meanrank,'%.6f' % meanReciprocalRank,'%.6f' % medianRank
#         ]
# =============================================================================
    
# =============================================================================
#     writeList = [filename, 
#         'testSet', '%.6f' % hit10Test, '%.6f' % meanrankTest, 
#         'one_to_one_head', '%.6f' % hit10OneToOneHead, '%.6f' % meanrankOneToOneHead, 
#         'one_to_many_head', '%.6f' % hit10OneToManyHead, '%.6f' % meanrankOneToManyHead, 
#         'many_to_one_head', '%.6f' % hit10ManyToOneHead, '%.6f' % meanrankManyToOneHead, 
#         'many_to_many_head', '%.6f' % hit10ManyToManyHead, '%.6f' % meanrankManyToManyHead,
#         'one_to_one_tail', '%.6f' % hit10OneToOneTail, '%.6f' % meanrankOneToOneTail, 
#         'one_to_many_tail', '%.6f' % hit10OneToManyTail, '%.6f' % meanrankOneToManyTail, 
#         'many_to_one_tail', '%.6f' % hit10ManyToOneTail, '%.6f' % meanrankManyToOneTail, 
#         'many_to_many_tail', '%.6f' % hit10ManyToManyTail, '%.6f' % meanrankManyToManyTail,]
# =============================================================================

    # Write the result into file
    if args.gray ==1:
        if not os.path.exists(os.path.join('/scratch/symin95/TransINTOutputs/result/',  args.which_dataset + '/'+args.dataset)):
            os.makedirs(os.path.join('/scratch/symin95/TransINTOutputs/result/',  args.which_dataset + '/'+args.dataset))
        with open(os.path.join('/scratch/symin95/TransINTOutputs/result/',  args.which_dataset + '/'+args.dataset + '.txt'), 'a') as fw:
            fw.write('\t'.join(writeList) + '\n')
    else:
        if not os.path.exists(os.path.join('./result/',  args.which_dataset + '/'+args.dataset)):
            os.makedirs(os.path.join('./result/',  args.which_dataset + '/'+args.dataset))
        with open(os.path.join('./result/',  args.which_dataset + '/'+args.dataset + '.txt'), 'a') as fw:
            fw.write('\t'.join(writeList) + '\n')
