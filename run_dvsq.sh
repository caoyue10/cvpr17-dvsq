nohup python train_script.py "0.1" "300" "5000" "0.0001" "4" "0.0" "81" "0" >> "snapshot_lr_0.1_cqlambda_0.0001_subspace_4_margin_0.0_partlabel_81_iter_5000_output_300.out"
nohup python validation_script.py "cos_softmargin_multi_label_lr_0.1_cqlambda_0.0001_subspace_4_margin_0.0_partlabel_81_iter_5000_output_300_.npy" "0" >> "valshot_lr_0.1_cqlambda_0.0001_subspace_4_margin_0.0_partlabel_81_iter_5000_output_300.out" 

