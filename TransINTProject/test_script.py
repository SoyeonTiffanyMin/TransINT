#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 09:28:30 2019

@author: TiffMin
"""
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


from model import *
torch.cuda.set_device(3)
mod = model.TransINTModel(config, rank_file, r_id2_vrr, equiv_flip_dict)
PATH = '/scratch/symin95/TransINTOutputs/model/KALE/FB122/dim_50_hr_gap1/l_0.01_es_1000_L_1_em_50_nb_100_n_1000_m_2.0_f_0_mo_0.9_s_0_op_1_lo_0/TransINT.ckpt'
mod=torch.load(PATH)
mod.eval()
mod =mod.cuda()

mod.vvr_bundle_heads_embs