#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-15 18:58:47
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import to_cuda

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

def projection_transH(original, norm):
	# numpy version
	return original - np.sum(original * norm, axis=1, keepdims=True) * norm

def projection_transH_pytorch(original, norm):
	return original - torch.sum(original * norm, dim=1, keepdim=True) * norm

def projection_transR_pytorch(original, proj_matrix):
	ent_embedding_size = original.shape[1]
	rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size
	original = original.view(-1, ent_embedding_size, 1)
	proj_matrix = proj_matrix.view(-1, rel_embedding_size, ent_embedding_size)
	return torch.matmul(proj_matrix, original).view(-1, rel_embedding_size)

def projection_transD_pytorch_samesize(entity_embedding, entity_projection, relation_projection):
	return entity_embedding + torch.sum(entity_embedding * entity_projection, dim=1, keepdim=True) * relation_projection

#basis_of_H_in_rows is actually a list of bases
def projection_matrix_transINT(list_of_basis_of_H_in_rows):
    #embed_size = list_of_basis_of_H_in_rows[0].shape[1]
    #return_proj = torch.randn(len(list_of_basis_of_H_in_rows), embed_size ,embed_size)
    list_proj = []
    for i, basis_of_H_in_rows in enumerate(list_of_basis_of_H_in_rows):
        basis_of_H_in_columns = basis_of_H_in_rows.t()
        A = basis_of_H_in_columns; AT = basis_of_H_in_rows
        ATA_inverse = torch.inverse(torch.mm(AT,A))
        proj = torch.mm(torch.mm(A, ATA_inverse), AT)
        #return_proj[i] = proj
        list_proj.append(to_cuda(proj.type(torch.FloatTensor)))
    return list_proj

def batch_projection_colvec_transINT(column_vectors, list_of_basis_of_H_in_rows):
    list_proj = projection_matrix_transINT(list_of_basis_of_H_in_rows)
    return_tensor = floatTensor(column_vectors.shape[1], column_vectors.shape[0])
    for i in range(column_vectors.shape[1]):
        single_col_vec = column_vectors[:,i].view(-1,1)
        return_tensor[i] = torch.mm(to_cuda(list_proj[i]), to_cuda(single_col_vec)).t()
    return return_tensor #return tensor of dim [num_ent x embed_dim] (embed vectors in row, not column)

def batch_projection_colvec_transINT_given_proj_mat_list(column_vectors, list_proj):
    list_proj = list_proj
    return_tensor = floatTensor(column_vectors.shape[1], column_vectors.shape[0])
    #print("return_tensor:", return_tensor)
    for i in range(column_vectors.shape[1]):
        single_col_vec = column_vectors[:,i].view(-1,1)
        #print("single_col_vec :", single_col_vec) 
        #print("list_proj[i] :", list_proj[i])
        return_tensor[i] = torch.mm(to_cuda(list_proj[i]), to_cuda(single_col_vec)).t()
    return return_tensor #return tensor of dim [num_ent x embed_dim] (embed vectors in row, not column)


def single_projection_colvec_transINT(column_vector, len_one_list_of_basis_of_H_in_rows):
    #column_vector = column_vector.view(-1, 1)
    list_proj = projection_matrix_transINT(len_one_list_of_basis_of_H_in_rows)
    #print("list proj :", list_proj)
    #print("column_vector :", to_cuda(column_vector).type(torch.FloatTensor))
    return_tensor = torch.mm(list_proj[0].type(torch.FloatTensor), to_cuda(column_vector).type(torch.FloatTensor)).t()#.squeeze(0)
    #print("return tensor:", return_tensor)
    return return_tensor #return tensor of dim [num_ent x embed_dim] (embed vectors in row, not column)