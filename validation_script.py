#                   model_path                gpu console
# validation.sh lr_0.01_output_300_iter_1000.npy 0 (1)
import numpy as np
import scipy.io as sio
import warnings
import dataset
import net_val as model
import sys

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

model_weight = sys.argv[1]
gpu = sys.argv[2]
console = len(sys.argv) > 3
output_dim = int(model_weight.split('_')[-2])
subspace_num = int(model_weight.split('_')[9])


config = {
    'device': '/gpu:' + gpu,
    'gpu_usage': 11,#G
    'max_iter': 5000,
    'batch_size': 100,
    'moving_average_decay': 0.9999,      # The decay to use for the moving average. 
    'decay_step': 500,          # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.1,   # Learning rate decay factor.
    'learning_rate': 0.001,       # Initial learning rate img.

    'output_dim': output_dim,
    'console_log': console,

    'R': 5000,
    # trained model for validation
    'model_weights': model_weight,
    
    'img_model': 'alexnet',
    'stage': 'validation',
    'loss_type': 'cos_softmargin_multi_label',

    'margin_param': 0.7,
    'wordvec_dict': "./data/nuswide_81/nuswide_wordvec.txt",
    'part_ids_dict': "./data/nuswide_81/train_81_ids.txt",
    'partlabel': 81,

    # only finetune last layer
    'finetune_all': False,

    'max_iter_update_b': 3,
    'max_iter_update_Cb': 1,
    'cq_lambda': 0.1,
    'code_batch_size': 50 * 14,
    'n_subspace': subspace_num,
    'n_subcenter': 256,   

    'n_train': 10000,
    'n_database': 168692,
    'n_query': 5000,

    'label_dim': 81,
    'img_tr': "./data/nuswide_81/train.txt",
    'img_te': "./data/nuswide_81/test.txt",
    'img_db': "./data/nuswide_81/database.txt",
    'save_dir': "./models/",
}

query_img, database_img = dataset.import_validation(config)

model.validation(database_img, query_img, config)
