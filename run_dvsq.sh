#!/bin/bash

#                           lr      output  iter    q_lamb  n_sub   margin  part_label  gpu
python train_script.py      0.02    300     5000    0.0001  4       0.7     10          0
python validation_script.py ./cos_softmargin_multi_label_lr_0.02_cqlambda_0.0001_subspace_4_margin_0.7_partlabel_10_iter_5_output_300_.npy 1
