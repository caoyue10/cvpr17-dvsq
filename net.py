##################################################################################
# Deep Visual-Semantic Quantization for Efficient Image Retrieval                #
# Authors: Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu                    #
# Contact: caoyue10@gmail.com                                                    #
##################################################################################

import os
import sys
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
from datetime import datetime
from math import ceil
import random
from util import ProgressBar, Dataset, MAPs, MAPs_CQ
from sklearn.cluster import MiniBatchKMeans

class DVSQ(object):
    def __init__(self, config):
        ### Initialize setting
        print ("initializing")
        np.set_printoptions(precision=4)
        self.stage = config['stage']
        self.device = config['device']
        self.output_dim = config['output_dim']
        self.n_class = config['label_dim']
        
        self.subspace_num = config['n_subspace']
        self.subcenter_num = config['n_subcenter']
        self.code_batch_size = config['code_batch_size']
        self.cq_lambda = config['cq_lambda']
        self.max_iter_update_Cb = config['max_iter_update_Cb']
        self.max_iter_update_b = config['max_iter_update_b']
        #self.centers_device = config['centers_device']
        
        self.batch_size = config['batch_size']
        self.max_iter = config['max_iter']
        self.img_model = config['img_model']
        self.loss_type = config['loss_type']
        self.console_log = (config['console_log'] == 1)
        self.learning_rate = config['learning_rate']
        self.learning_rate_decay_factor = config['learning_rate_decay_factor']
        self.decay_step = config['decay_step']
        
        self.finetune_all = config['finetune_all']

        self.margin_param = config['margin_param']
        self.wordvec_dict = config['wordvec_dict']
        self.partlabel = config['partlabel']
        ### Format as 'path/to/save/dir/lr_{$0}_output_dim{$1}_iter_{$2}'
        self.save_dir = config['save_dir'] + self.loss_type + '_lr_' + str(self.learning_rate) + '_cqlambda_'+ str(self.cq_lambda) + '_subspace_' + str(self.subspace_num) + '_margin_' + str(self.margin_param) + '_partlabel_' + str(self.partlabel) + '_iter_' + str(self.max_iter) + '_output_' + str(self.output_dim) + '_'

        ### Setup session
        print ("launching session")
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)

        ### Create variables and placeholders
        
        with tf.device(self.device):
            self.img = tf.placeholder(tf.float32, [self.batch_size, 256, 256, 3])
            self.img_label = tf.placeholder(tf.float32, [self.batch_size, self.n_class])

            self.img_last_layer, self.img_output, self.C = \
                self.load_model(config['model_weights'])
                
            ### Centers shared in different modalities (image & text)
            ### Binary codes for different modalities (image & text)
            self.img_output_all = tf.placeholder(tf.float32, [None, self.output_dim])
            self.img_b_all = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])

            self.b_img = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])
            self.ICM_m = tf.placeholder(tf.int32, [])
            self.ICM_b_m = tf.placeholder(tf.float32, [None, self.subcenter_num])
            self.ICM_b_all = tf.placeholder(tf.float32, [None, self.subcenter_num * self.subspace_num])
            self.ICM_X = tf.placeholder(tf.float32, [self.code_batch_size, self.output_dim])
            self.ICM_C_m = tf.slice(self.C, [self.ICM_m * self.subcenter_num, 0], [self.subcenter_num, self.output_dim])
            self.ICM_X_residual = tf.add(tf.sub(self.ICM_X, tf.matmul(self.ICM_b_all, self.C)), tf.matmul(self.ICM_b_m, self.ICM_C_m))
            ICM_X_expand = tf.expand_dims(self.ICM_X_residual, 1)
            ICM_C_m_expand = tf.expand_dims(self.ICM_C_m, 0)
            # N*sc*D  *  D*n
            word_dict = tf.constant(np.loadtxt(self.wordvec_dict), dtype=tf.float32)
            ICM_word_dict = tf.reshape(tf.matmul(tf.reshape(tf.sub(ICM_X_expand, ICM_C_m_expand), [self.code_batch_size*self.subcenter_num, self.output_dim]), tf.transpose(word_dict)), [self.code_batch_size, self.subcenter_num, self.n_class])
            ICM_sum_squares = tf.reduce_sum(tf.square(ICM_word_dict), reduction_indices = 2)
            ICM_best_centers = tf.argmin(ICM_sum_squares, 1)
            self.ICM_best_centers_one_hot = tf.one_hot(ICM_best_centers, self.subcenter_num, dtype = tf.float32)

            self.global_step = tf.Variable(0, trainable=False)
            self.train_op = self.apply_loss_function(self.global_step)
            self.sess.run(tf.initialize_all_variables())
        return

    def load_model(self, img_model_weights):
        if self.img_model == 'alexnet':
            img_output = self.img_alexnet_layers(img_model_weights)
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)
        return img_output

    def img_alexnet_layers(self, model_weights):
        self.deep_param_img = {}
        self.train_layers = []
        self.train_last_layer = []
        print ("loading img model")
        net_data = np.load(model_weights).item()
        
        # swap(2,1,0)
        reshaped_image = tf.cast(self.img, tf.float32)
        tm = tf.Variable([[0,0,1],[0,1,0],[1,0,0]],dtype=tf.float32)
        reshaped_image = tf.reshape(reshaped_image,[self.batch_size * 256 * 256, 3])
        reshaped_image = tf.matmul(reshaped_image,tm)
        reshaped_image = tf.reshape(reshaped_image,[self.batch_size, 256 , 256, 3])
        
        IMAGE_SIZE = 227
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        ### Randomly crop a [height, width] section of each image
        distorted_image = tf.pack([tf.random_crop(tf.image.random_flip_left_right(each_image), [height, width, 3]) for each_image in tf.unpack(reshaped_image)])

        ### Zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
            distorted_image = distorted_image - mean

        ### Conv1
        ### Output 96, kernel 11, stride 4
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(net_data['conv1'][0], name='weights')
            conv = tf.nn.conv2d(distorted_image, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(net_data['conv1'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv1'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Pool1
        self.pool1 = tf.nn.max_pool(self.conv1,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool1')

        ### LRN1
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        ### Conv2
        ### Output 256, pad 2, kernel 5, group 2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(net_data['conv2'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.lrn1)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            
            biases = tf.Variable(net_data['conv2'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv2'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Pool2
        self.pool2 = tf.nn.max_pool(self.conv2,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool2')

        ### LRN2
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        ### Conv3
        ### Output 384, pad 1, kernel 3
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(net_data['conv3'][0], name='weights')
            conv = tf.nn.conv2d(self.lrn2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv3'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Conv4
        ### Output 384, pad 1, kernel 3, group 2
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(net_data['conv4'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv3)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv4'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv4'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Conv5
        ### Output 256, pad 1, kernel 3, group 2
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(net_data['conv5'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv4)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv5'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv5'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Pool5
        self.pool5 = tf.nn.max_pool(self.conv5,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool5')

        ### FC6
        ### Output 4096
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.Variable(net_data['fc6'][0], name='weights')
            fc6b = tf.Variable(net_data['fc6'][1], name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            self.fc5 = pool5_flat
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
            self.fc6o = tf.nn.relu(fc6l)
            self.deep_param_img['fc6'] = [fc6w, fc6b]
            self.train_layers += [fc6w, fc6b]

        ### FC7
        ### Output 4096
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(net_data['fc7'][0], name='weights')
            fc7b = tf.Variable(net_data['fc7'][1], name='biases')
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
            fc7lo = tf.nn.bias_add(tf.matmul(self.fc6o, fc7w), fc7b)            
            self.fc7o = tf.nn.relu(fc7lo)
            self.deep_param_img['fc7'] = [fc7w, fc7b]
            self.train_layers += [fc7w, fc7b]

        ### FC8
        ### Output output_dim
        with tf.name_scope('fc8') as scope:
            ### Differ train and val stage by 'fc8' as key
            if 'fc8' in net_data:
                fc8w = tf.Variable(net_data['fc8'][0], name='weights')
                fc8b = tf.Variable(net_data['fc8'][1], name='biases')
            else:
                fc8w = tf.Variable(tf.random_normal([4096, self.output_dim],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
                fc8b = tf.Variable(tf.constant(0.0, shape=[self.output_dim],
                                               dtype=tf.float32), name='biases')
            fc8l = tf.nn.bias_add(tf.matmul(self.fc7, fc8w), fc8b)
            self.fc8 = tf.nn.tanh(fc8l)
            fc8lo = tf.nn.bias_add(tf.matmul(self.fc7o, fc8w), fc8b)
            self.fc8o = tf.nn.tanh(fc8lo)
            self.deep_param_img['fc8'] = [fc8w, fc8b]
            self.train_last_layer += [fc8w, fc8b]
            self.test3 = fc8w
            self.test1 = fc8b
        
        ### load centers
        if 'C' in net_data:
            self.centers = tf.Variable(net_data['C'], name='weights')
        else:
            self.centers = tf.Variable(tf.random_uniform([self.subspace_num * self.subcenter_num, self.output_dim],
                                    minval = -1, maxval = 1, dtype = tf.float32, name = 'centers'))

        self.deep_param_img['C'] = self.centers

        print("img modal loading finished")
        ### Return outputs
        return self.fc8, self.fc8o, self.centers

    def save_model(self, model_file=None):
        if model_file == None:
            model_file = self.save_dir
        model = {}
        for layer in self.deep_param_img:
            model[layer] = self.sess.run(self.deep_param_img[layer])
        print ("saving model to %s" % model_file)
        np.save(model_file, np.array(model))
        return

    def apply_loss_function(self, global_step):
        ### loss function
        if self.loss_type == 'cos_margin_multi_label':
            assert self.output_dim == 300
            word_dict = tf.constant(np.loadtxt(self.wordvec_dict), dtype=tf.float32)
            margin_param = tf.constant(self.margin_param, dtype=tf.float32)

            # N: batchsize, L: label_dim, D: 300
            # img_label: N * L
            # word_dic: L * D
            # v_label: N * L * D
            v_label = tf.mul(tf.expand_dims(self.img_label, 2), tf.expand_dims(word_dict, 0))
            # img_last: N * D
            # ip_1: N * L
            ip_1 = tf.reduce_sum(tf.mul(tf.expand_dims(self.img_last_layer, 1), v_label), 2)
            # mod_1: N * L
            v_label_mod = tf.mul(tf.expand_dims(tf.ones([self.batch_size, self.n_class]), 2), tf.expand_dims(word_dict, 0))
            mod_1 = tf.sqrt(tf.mul(tf.expand_dims(tf.reduce_sum(tf.square(self.img_last_layer), 1), 1), tf.reduce_sum(tf.square(v_label_mod), 2)))
            #mod_1 = tf.select(tf.less(mod_1_1, tf.constant(0.0000001)), tf.ones([self.batch_size, self.n_class]), mod_1_1)
            # cos_1: N * L
            cos_1 = tf.div(ip_1, mod_1)

            ip_2 = tf.matmul(self.img_last_layer, word_dict, transpose_b=True)
            # multiply ids to inner product
            #ip_2 = tf.mul(ip_2_1, ids_dict)
            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            mod_2_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.img_last_layer)), reduce_shaper(tf.square(word_dict)), transpose_b=True))
            mod_2 = tf.select(tf.less(mod_2_2, tf.constant(0.0000001)), tf.ones([self.batch_size, self.n_class]), mod_2_2)
            # cos_2: N * L
            cos_2 = tf.div(ip_2, mod_2)

            # cos - cos: N * L * L
            cos_cos_1 = tf.sub(margin_param, tf.sub(tf.expand_dims(cos_1, 2), tf.expand_dims(cos_2, 1)))
            # we need to let the wrong place be 0
            cos_cos = tf.mul(cos_cos_1, tf.expand_dims(self.img_label, 2))

            cos_loss = tf.reduce_sum(tf.maximum(tf.constant(0, dtype=tf.float32), cos_cos))
            self.cos_loss = tf.div(cos_loss, tf.mul(tf.constant(self.n_class, dtype=tf.float32), tf.reduce_sum(self.img_label)))

            self.test1 = cos_cos
            self.test2 = cos_1
            self.test3 = cos_2
            self.test00 = tf.reduce_sum(tf.cast(tf.not_equal(cos_cos, tf.zeros([self.batch_size, self.n_class, self.n_class])), tf.int32))
            self.test0 = tf.mul(tf.constant(self.n_class, dtype=tf.float32), tf.reduce_sum(self.img_label))
            self.check0 = tf.check_numerics(cos_cos, "cos_cos")
            self.check1 = tf.check_numerics(cos_1, "cos_1")
            self.check2 = tf.check_numerics(cos_2, "cos_2")
        elif self.loss_type == 'cos_softmargin_multi_label':
            assert self.output_dim == 300
            word_dict = tf.constant(np.loadtxt(self.wordvec_dict), dtype=tf.float32)
            #margin_param = tf.constant(self.margin_param, dtype=tf.float32)

            # N: batchsize, L: label_dim, D: 300
            # img_label: N * L
            # word_dic: L * D
            # v_label: N * L * D
            v_label = tf.mul(tf.expand_dims(self.img_label, 2), tf.expand_dims(word_dict, 0))
            # img_last: N * D
            # ip_1: N * L
            ip_1 = tf.reduce_sum(tf.mul(tf.expand_dims(self.img_last_layer, 1), v_label), 2)
            # mod_1: N * L
            v_label_mod = tf.mul(tf.expand_dims(tf.ones([self.batch_size, self.n_class]), 2), tf.expand_dims(word_dict, 0))
            mod_1 = tf.sqrt(tf.mul(tf.expand_dims(tf.reduce_sum(tf.square(self.img_last_layer), 1), 1), tf.reduce_sum(tf.square(v_label_mod), 2)))
            #mod_1 = tf.select(tf.less(mod_1_1, tf.constant(0.0000001)), tf.ones([self.batch_size, self.n_class]), mod_1_1)
            # cos_1: N * L
            cos_1 = tf.div(ip_1, mod_1)

            ip_2 = tf.matmul(self.img_last_layer, word_dict, transpose_b=True)
            # multiply ids to inner product
            #ip_2 = tf.mul(ip_2_1, ids_dict)
            def reduce_shaper(t):
                return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])
            mod_2_2 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(self.img_last_layer)), reduce_shaper(tf.square(word_dict)), transpose_b=True))
            mod_2 = tf.select(tf.less(mod_2_2, tf.constant(0.0000001)), tf.ones([self.batch_size, self.n_class]), mod_2_2)
            # cos_2: N * L
            cos_2 = tf.div(ip_2, mod_2)
            
            # word_dic: L * D
            # ip_3: L * L
            # compute soft margin
            ip_3 = tf.matmul(word_dict, word_dict, transpose_b=True)
            # use word_dic to avoid 0 in /
            mod_3 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(word_dict)), reduce_shaper(tf.square(word_dict)), transpose_b=True))
            margin_param = tf.sub(tf.constant(1.0, dtype=tf.float32), tf.div(ip_3, mod_3))

            # cos - cos: N * L * L
            cos_cos_1 = tf.sub(tf.expand_dims(margin_param, 0), tf.sub(tf.expand_dims(cos_1, 2), tf.expand_dims(cos_2, 1)))
            # we need to let the wrong place be 0
            cos_cos = tf.mul(cos_cos_1, tf.expand_dims(self.img_label, 2))

            cos_loss = tf.reduce_sum(tf.maximum(tf.constant(0, dtype=tf.float32), cos_cos))
            self.cos_loss = tf.div(cos_loss, tf.mul(tf.constant(self.n_class, dtype=tf.float32), tf.reduce_sum(self.img_label)))

            self.test1 = cos_cos
            self.test2 = cos_1
            self.test3 = cos_2
        
        
        
        self.precq_loss_img = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(self.img_last_layer, tf.matmul(self.b_img, self.C))), 1))
        word_dict = tf.constant(np.loadtxt(self.wordvec_dict), dtype=tf.float32)
        self.cq_loss_img = tf.reduce_mean(tf.reduce_sum(tf.square(tf.matmul(tf.sub(self.img_last_layer, tf.matmul(self.b_img, self.C)), tf.transpose(word_dict))), 1))
        self.q_lambda = tf.Variable(self.cq_lambda, name='cq_lambda')
        self.cq_loss = tf.mul(self.q_lambda, self.cq_loss_img)
        self.loss = tf.add(self.cos_loss, self.cq_loss)

        ### Last layer has a 10 times learning rate
        self.lr = tf.train.exponential_decay(self.learning_rate, global_step, self.decay_step, self.learning_rate_decay_factor, staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        grads_and_vars = opt.compute_gradients(self.loss, self.train_layers+self.train_last_layer)
        fcgrad, _ = grads_and_vars[-2]
        fbgrad, _ = grads_and_vars[-1]

        if self.finetune_all:
            return opt.apply_gradients([(grads_and_vars[0][0], self.train_layers[0]),
                                        (grads_and_vars[1][0]*2, self.train_layers[1]),
                                        (grads_and_vars[2][0], self.train_layers[2]),
                                        (grads_and_vars[3][0]*2, self.train_layers[3]),
                                        (grads_and_vars[4][0], self.train_layers[4]),
                                        (grads_and_vars[5][0]*2, self.train_layers[5]),
                                        (grads_and_vars[6][0], self.train_layers[6]),
                                        (grads_and_vars[7][0]*2, self.train_layers[7]),
                                        (grads_and_vars[8][0], self.train_layers[8]),
                                        (grads_and_vars[9][0]*2, self.train_layers[9]),
                                        (grads_and_vars[10][0], self.train_layers[10]),
                                        (grads_and_vars[11][0]*2, self.train_layers[11]),
                                        (grads_and_vars[12][0], self.train_layers[12]),
                                        (grads_and_vars[13][0]*2, self.train_layers[13]),
                                        (fcgrad*10, self.train_last_layer[0]),
                                        (fbgrad*20, self.train_last_layer[1])], global_step=global_step)
        else:
            return opt.apply_gradients([(fcgrad*10, self.train_last_layer[0]),
                                        (fbgrad*20, self.train_last_layer[1])], global_step=global_step)

    def initial_centers(self, img_output):
        C_init = np.zeros([self.subspace_num * self.subcenter_num, self.output_dim])
        print "#DVSQ train# initilizing Centers"
        all_output = img_output
        for i in xrange(self.subspace_num):
            kmeans = MiniBatchKMeans(n_clusters=self.subcenter_num).fit(all_output[:, i * self.output_dim / self.subspace_num: (i + 1) * self.output_dim / self.subspace_num])
            C_init[i * self.subcenter_num: (i + 1) * self.subcenter_num, i * self.output_dim / self.subspace_num: (i + 1) * self.output_dim / self.subspace_num] = kmeans.cluster_centers_
            print "step: ", i, " finish"
        return C_init

    def update_centers(self, img_dataset):
        '''
        Optimize:
            self.C = (U * hu^T + V * hv^T) (hu * hu^T + hv * hv^T)^{-1}
            self.C^T = (hu * hu^T + hv * hv^T)^{-1} (hu * U^T + hv * V^T)
            but all the C need to be replace with C^T :
            self.C = (hu * hu^T + hv * hv^T)^{-1} (hu^T * U + hv^T * V)
        '''
        old_C_value = self.sess.run(self.C)
        
        h = self.img_b_all
        U = self.img_output_all
        smallResidual = tf.constant(np.eye(self.subcenter_num * self.subspace_num, dtype = np.float32) * 0.001)
        Uh = tf.matmul(tf.transpose(h), U)
        hh = tf.add(tf.matmul(tf.transpose(h), h), smallResidual)
        compute_centers = tf.matmul(tf.matrix_inverse(hh), Uh)
        
        update_C = self.C.assign(compute_centers)
        C_value = self.sess.run(update_C, feed_dict = {
            self.img_output_all: img_dataset.output, 
            self.img_b_all: img_dataset.codes,
            })
        
        C_sums = np.sum(np.square(C_value), axis=1)
        C_zeros_ids = np.where(C_sums < 1e-8)
        C_value[C_zeros_ids, :] = old_C_value[C_zeros_ids, :]
        self.sess.run(self.C.assign(C_value))

        print 'updated C is:'
        print C_value
        print "non zeros:"
        print len(np.where(np.sum(C_value, 1) != 0)[0])

    def update_codes_ICM(self, output, code):
        '''
        Optimize:
            min || output - self.C * codes ||
            min || output - codes * self.C ||
        args:
            output: [n_train, n_output]
            self.C: [n_subspace * n_subcenter, n_output]
                [C_1, C_2, ... C_M]
            codes: [n_train, n_subspace * n_subcenter]
        '''

        code = np.zeros(code.shape)
        
        for iterate in xrange(self.max_iter_update_b):
            start = time.time()
            time_init = 0.0
            time_compute_centers = 0.0
            time_append = 0.0
            
            sub_list = [i for i in range(self.subspace_num)]
            random.shuffle(sub_list)
            for m in sub_list:
                best_centers_one_hot_val = self.sess.run(self.ICM_best_centers_one_hot, feed_dict = {
                    self.ICM_b_m: code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num],
                    self.ICM_b_all: code,
                    self.ICM_m: m,
                    self.ICM_X: output,
                })

                code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num] = best_centers_one_hot_val
        return code
    
    def update_codes_batch(self, dataset, batch_size):
        '''
        update codes in batch size
        '''
        total_batch = int(ceil(dataset.n_samples / batch_size))
        print "start update codes in batch size ", batch_size

        dataset.finish_epoch()
        
        for i in xrange(total_batch):
            print "Iter ", i, "of ", total_batch
            output_val, code_val = dataset.next_batch_output_codes(batch_size)
            print output_val, code_val
            codes_val = self.update_codes_ICM(output_val, code_val)
            print np.sum(np.sum(codes_val, 0) != 0)
            dataset.feed_batch_codes(batch_size, codes_val)

        print "update_code wrong:"
        print np.sum(np.sum(dataset.codes, 1) != 4)
        
        print "######### update codes done ##########"

    def train_cq(self, img_dataset):
        print ("%s #train# start training" % datetime.now())
        epoch = 0
        epoch_iter = int(ceil(img_dataset.n_samples / self.batch_size))
        
        for train_iter in xrange(self.max_iter):
            images, labels, codes = img_dataset.next_batch(self.batch_size)
            start_time = time.time()
            
            if epoch > 0:
                assign_lambda = self.q_lambda.assign(self.cq_lambda)
            else:
                assign_lambda = self.q_lambda.assign(0.0)
            self.sess.run([assign_lambda])

            _, cos_loss, cq_loss, fc8_value, test1, test2, test3, lr, output = self.sess.run([self.train_op, self.cos_loss, self.cq_loss, self.deep_param_img['fc8'], self.test1, self.test2, self.test3, self.lr, self.img_last_layer],
                                    feed_dict={self.img: images,
                                               self.img_label: labels,
                                               self.b_img: codes})
            img_dataset.feed_batch_output(self.batch_size, output)
            duration = time.time() - start_time
            
            # every epoch: update codes and centers
            if train_iter % (2*epoch_iter) == 0 and train_iter != 0:
                if epoch == 0:
                    with tf.device(self.device):
                        for i in xrange(self.max_iter_update_Cb):
                            print "#DVSQ Train# initialize centers in ", i, " iter"
                            self.sess.run(self.C.assign(self.initial_centers(img_dataset.output)))
                            
                        print "#DVSQ Train# initialize centers done!!!"
                epoch = epoch + 1
                for i in xrange(self.max_iter_update_Cb):
                    print "#DVSQ Train# update codes and centers in ", i, " iter"
                    self.update_codes_batch(img_dataset, self.code_batch_size)
                    self.update_centers(img_dataset)
            
            print("%s #train# step %4d, lr %.8f, cosine margin loss = %.4f, cq loss = %.4f, %.1f sec/batch" % (datetime.now(), train_iter+1, lr, cos_loss, cq_loss, duration))
                
        print ("%s #traing# finish training" % datetime.now())
        self.save_model()
        print ("model saved")

    
def train(train_img, config):
    model = DVSQ(config)
    img_dataset = Dataset(train_img, config['output_dim'], config['n_subspace'] * config['n_subcenter'])
    model.train_cq(img_dataset)
    return model.save_dir
