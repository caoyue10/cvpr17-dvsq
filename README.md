# aaai16-dqn

This is the Tensorflow (Version 0.11) implementation of AAAI-16 paper "Deep Quantization Networks for Efficient Image Retrieval". The descriptions of files in this directory are listed below:

- `net.py`: contains the main implementation (network structure, loss function, optimization procedure and etc.) of the proposed approach `dqn`.
- `net_val.py`: contains the implementation of `dqn` for evaluation.
- `util.py`: contains the implementation of `Dataset`, `MAP` and `ProcessBar`.
- `train_script_dqn.py`: gives an example to show how to train `dqn` model. 
- `validation_script_dqn.py`: gives an example to show how to evaluate the trained quantization model.
- `run_dqn.sh`: gives an example to show the full procedure of training and evaluating the proposed approach `dqn`.

Data Preparation
---------------
In `data/nuswide_21/train.txt`, we give an example to show how to prepare image training data. In `data/nuswide_21/test.txt` and `data/nuswide_21/database.txt`, the list of testing and database images could be processed during predicting procedure.

Training Model and Predicting
---------------
The `bvlc_reference_caffenet` is used as the pre-trained model. If the NUS\_WIDE dataset and pre-trained caffemodel is prepared, the example can be run with the following command:
```
"./run_dqn.sh"
```

Citation
---------------

    @inproceedings{DBLP:conf/aaai/CaoL0ZW16,
      author    = {Yue Cao and
                   Mingsheng Long and
                   Jianmin Wang and
                   Han Zhu and
                   Qingfu Wen},
      title     = {Deep Quantization Network for Efficient Image Retrieval},
      booktitle = {Proceedings of the Thirtieth {AAAI} Conference on Artificial Intelligence,
                   February 12-17, 2016, Phoenix, Arizona, {USA.}},
      pages     = {3457--3463},
      year      = {2016},
      crossref  = {DBLP:conf/aaai/2016},
      url       = {http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12040},
      timestamp = {Thu, 21 Apr 2016 19:28:00 +0200},
      biburl    = {http://dblp.uni-trier.de/rec/bib/conf/aaai/CaoL0ZW16},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
