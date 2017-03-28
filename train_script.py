#          lr output iter gpu console
# train.sh 0.01 300 5000 0.1 3 4 0.7 100 0 (1)
import numpy as np
import scipy.io as sio
import warnings
import dataset
import net as model
import sys

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

### Define input arguments
lr = float(sys.argv[1])
output_dim = int(sys.argv[2])
iter_num = int(sys.argv[3])
cq_lambda = float(sys.argv[4])
#update_b = int(sys.argv[5])
subspace_num = int(sys.argv[5])
margin_param = float(sys.argv[6])
part_label = int(sys.argv[7])
gpu = sys.argv[8]

console = len(sys.argv) > 10

config = {
    'device': '/gpu:' + gpu,
    'gpu_usage': 11,#G
    'max_iter': iter_num,
    'batch_size': 256,
    'moving_average_decay': 0.9999,      # The decay to use for the moving average. 
    'decay_step': 500,                   # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.5,   # Learning rate decay factor.
    'learning_rate': lr,                 # Initial learning rate img.
    'console_log': console,

    'output_dim': output_dim,

    'R': 5000,
    'model_weights': 'models/reference_pretrain.npy',

    'img_model': 'alexnet',
    'stage': 'train',
    'loss_type': 'cos_softmargin_multi_label',
    
    'margin_param': margin_param,
    'wordvec_dict': "./data/nuswide_81/nuswide_wordvec.txt",
    'part_ids_dict': "./data/nuswide_81/train_"+str(part_label)+"_ids.txt",
    'partlabel': part_label,
    
    # only finetune last layer
    'finetune_all': True,
    
    ## CQ params
    'max_iter_update_b': 3,
    'max_iter_update_Cb': 1,
    'cq_lambda': cq_lambda,
    'code_batch_size': 500,
    'n_subspace': subspace_num,
    'n_subcenter': 256,

    'n_train': 10000 * (part_label/81),
    'n_database': 218491,
    'n_query': 5000,

    'label_dim': 81,
    'img_tr': "./data/nuswide_81/train.txt",
    'img_te': "./data/nuswide_81/test.txt",
    'img_db': "./data/nuswide_81/database.txt",
    'save_dir': "./",
}

import time
t = time.time()
train_img = dataset.import_train(config)
print time.time() - t

model_dq = model.train(train_img, config)
