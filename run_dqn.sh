nohup python train_script_dqn.py "0.005" "300" "5000" "0.0001" "4" "0.0" "81" "2" >> "snapshot_lr_0.005_cqlambda_0.0001_subspace_4_margin_0.0_partlabel_81_iter_5000_output_300.out"
nohup python validation_script_dqn.py "cos_lr_0.005_cqlambda_0.0001_subspace_4_margin_0.0_partlabel_81_iter_5000_output_300_.npy" "2" >> "valshot_lr_0.005_cqlambda_0.0001_subspace_4_margin_0.0_partlabel_81_iter_5000_output_300.out" 

