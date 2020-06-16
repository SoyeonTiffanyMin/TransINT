# TransINT
Repository for AKBC 2020 Publication "TransINT: Embedding Implication Rules in Knowledge Graphs with Isomorphic Intersections of Linear Subspaces"
Link to paper: https://openreview.net/forum?id=shkmWLRBXH

The codes in this repository was built upon the codes of https://github.com/jimmywangheng/knowledge_representation_pytorch. While codes for the contents of the TransINT paper were originally written, the fundamental codes for knowledge graph embeddings (e.g. TransH) were taken directly from the repository above.

## Prerequisites 
Please install Python 3.6 and Pytorch

## How to Replicate Results

In order to replicate results, please first download the pre-processed data here: https://drive.google.com/file/d/1wBuG3zBkNe7L5i7S-eZBKx88VhTYeoHQ/view?usp=sharing
Unzip it and place the "dataset" folder right under "TransINTProject". 

### Results for Link Prediction on FB122
Run the following commands at the "TransINTProject" directory.
1. TransINT$$^G$$
```
python transINT_Bernoulli_pytorch.py  -l 0.003 -es 1000 -f 1 -n 1000 -em 100 -lr_decay 1  -c 0 -imp 1  -rank_file NULL_dim_100_hr_gap1_dict.p -d KALE/FB122/dim_100_hr_gap1_no_norm_ent_f1imp1   -norm_rel 0 -norm_ent 0 -norm_mat 0 -m 5  -L 1 -eval_cycle 10  -gray 1
```
(Optimal Configuration was - learning rate: 0.003, number of dimensions (d): 100, margin: 5, learning rate decay: 0.95, no normalization on entity vectors, relation vectors, and relation subspace bases, L1 (among L1 and L2 norms in calculating distances))

2. TransINT$$^{NG}$$
```
python transINT_Bernoulli_pytorch.py  -l 0.003 -es 1000 -f 1 -n 1000 -em 50 -lr_decay 0.95  -c 0 -imp 0  -rank_file NULL_dim_50_hr_gap1_dict.p -d KALE/FB122/dim_50_hr_gap1_no_norment_noimp  -gray 1  -norm_rel 0 -norm_ent 0 -norm_mat 0 -m 5  -L 1 -eval_cycle 10
```
(Optimal Configuration was : learning rate: 0.003, number of dimensions (d): 50, margin: 5, learning rate decay: 0.95, no normalization on entity vectors, relation vectors, and relation subspace bases, L1 (among L1 and L2 norms in calculating distances))

<!---### Results for Triple Classificaion on FB122
1. TransINT$^G$

(Optimal Configuration was :)

2. TransINT$^{NG}$
(Optimal Configuration was : ) -->


### Results for Link Prediction on NELL Sport
1. TransINT$$^G$$
```
python transINT_Bernoulli_pytorch.py  -l 0.003 -es 1000 -f 1 -n 1000 -em 100 -lr_decay 1  -c 0 -imp 1  -rank_file NULL_dim_100_hr_gap1_dict.p -d KALE/FB122/dim_100_hr_gap1_no_norm_ent_f1imp1   -norm_rel 0 -norm_ent 0 -norm_mat 0 -m 5  -L 1 -eval_cycle 10  -gray 1
```
(Optimal Configuration was - learning rate: 0.003, number of dimensions (d): 100, margin: 5, learning rate decay: 0.95, no normalization on entity vectors, relation vectors, and relation subspace bases, L1 (among L1 and L2 norms in calculating distances))

2. TransINT$$^{NG}$$
```
python transINT_Bernoulli_pytorch.py  -l 0.003 -es 1000 -f 1 -n 1000 -em 100 -lr_dey 0.95  -c 1 -imp 0  -rank_file three_2_NULL_dim_100_hr_gap1_dict.p -d KALE/Nell_sp/three_2_dim_100_hr_gap1_no_norment_no_normrel_imp0  -gray 1 -norm_rel 0 -norm_ent 0 -norm_mat 0 -m 5  -L 1 -which_dataset Nell_sp -eval_cycle 10
```
(Optimal Configuration was - learning rate: 0.003, number of dimensions (d): 100, margin: 5, learning rate decay: 0.95, no normalization on entity vectors, relation vectors, and relation subspace bases, L1 (among L1 and L2 norms in calculating distances))


### Results for Link Prediction on NELL Locations
1. TransINT$$^G$$
```
python transINT_Bernoulli_pytorch.py  -l 0.0075 -es 1000 -f 1 -n 1000 -em 50 -lr_decay 0.95  -c 1 -imp 1  -rank_file NULL_dim_50_hr_gap1_dict.p -d KALE/Nell_loc/dim_50_hr_gap1_no_norment_no_normrel_yes_imp  -gray 1 -norm_rel 0 -norm_ent 0 -norm_mat 0 -m 5  -L 1 -which_dataset Nell_loc -eval_cycle 10
```
(Optimal Configuration was - learning rate: 0.0075, number of dimensions (d): 50, margin: 5, learning rate decay: 0.95, no normalization on entity vectors, relation vectors, and relation subspace bases, L1 (among L1 and L2 norms in calculating distances))

2. TransINT$$^{NG}$$
```
python transINT_Bernoulli_pytorch.py  -l 0.001 -es 1000 -f 1 -n 1000 -em 50 -lr_decay 0.925  -c 1 -imp 0  -rank_file NULL_dim_50_hr_gap1_dict.p -d KALE/Nell_loc/dim_50_hr_gap1_no_norment_no_normrel  -gray 1 -norm_rel 0 -norm_ent 0 -norm_mat 0 -m 5  -L 1 -which_dataset Nell_loc -eval_cycle 10
```
(Optimal Configuration was - learning rate: 0.001, number of dimensions (d): 50, margin: 5, learning rate decay: 0.925, no normalization on entity vectors, relation vectors, and relation subspace bases, L1 (among L1 and L2 norms in calculating distances))

### How to Cite

@inproceedings{
min2020transint,
title={Trans{\{}INT{\}}: Embedding Implication Rules in Knowledge Graphs with Isomorphic Intersections of Linear Subspaces},
author={So Yeon Min and Preethi Raghavan and Peter Szolovits},
booktitle={Automated Knowledge Base Construction},
year={2020},
url={https://openreview.net/forum?id=shkmWLRBXH}
}

<script type="text/javascript" async

src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
