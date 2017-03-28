# cvpr17-dvsq

This is the Tensorflow (Version 0.11) implementation of CVPR-17 paper "Deep Visual-Semantic Quantization for Efficient Image Retrieval". The descriptions of files in this directory are listed below:

- `net.py`: contains the main implementation (network structure, loss function, optimization procedure and etc.) of the proposed approach `dvsq`.
- `net_val.py`: contains the implementation of `dvsq` for evaluation.
- `util.py`: contains the implementation of `Dataset`, `MAP` and `ProcessBar`.
- `train_script.py`: gives an example to show how to train `dvsq` model. 
- `validation_script.py`: gives an example to show how to evaluate the trained quantization model.
- `run_dvsq.sh`: gives an example to show the full procedure of training and evaluating the proposed approach `dvsq`.

Data Preparation
---------------
In `data/nuswide_81/train.txt`, we give an example to show how to prepare image training data. In `data/nuswide_81/test.txt` and `data/nuswide_81/database.txt`, the list of testing and database images could be processed during predicting procedure. In `data/nuswide_81/nuswide_wordvec.txt`, we have already prepared the word vectors of the labels extracted by [Word2Vec model](https://code.google.com/archive/p/word2vec/) pretrained on Google News Dataset.

Training Model and Predicting
---------------
The `bvlc_reference_caffenet` is used as the pre-trained model. If the NUS\_WIDE dataset and pre-trained caffemodel is prepared, the example can be run with the following command:
```
"./run_dvsq.sh"
```

Citation
---------------
    @inproceedings{conf/cvpr/CaoL0L17,
      author    = {Yue Cao and
                   Mingsheng Long and
                   Jianmin Wang and
                   Shichen Liu},
      title     = {Deep Visual-Semantic Quantization for Efficient Image Retrieval},
      booktitle = {2017 {IEEE} Conference on Computer Vision and Pattern Recognition,
          {CVPR} 2017, Honolulu, Hawaii, USA, July 21-26, 2017}
    }
