# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : pargs.py
# @Software: PyCharm
"""

import os
import torch
import argparse


def pargs():
    parser = argparse.ArgumentParser(description='AQG generation args')
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--clip_grad', type=float, default=0.6, help='gradient clipping')

    parser.add_argument("--d_emb", default=300, type=int)
    parser.add_argument("--d_h", default=256, type=int)
    parser.add_argument("--n_lstm_layers", default=1, type=int)
    parser.add_argument("--n_gnn_blocks", default=3, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument('--not_birnn', action='store_false', dest='birnn')

    parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    parser.add_argument('--att_type', default='affine', choices=['dot_prod', 'affine'])

    parser.add_argument("--max_num_op", default=20, type=int, help='maximum number of time steps used '
                                                                                 'in decoding')

    parser.add_argument('--use_small', action='store_true', help='use small data', dest='use_small')
    parser.add_argument('--not_shuffle', action='store_false', help='do not '
                                                                    'shuffle training data', dest='shuffle')

    parser.add_argument('--use_kb_constraint', action='store_true', dest='kb_constraint')

    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--word_normalize', action='store_true')

    parser.add_argument('--train_data', type=str, default=os.path.abspath('./data/processed_train.pkl'))
    parser.add_argument('--valid_data', type=str, default=os.path.abspath('./data/processed_valid.pkl'))
    parser.add_argument('--test_data', type=str, default=os.path.abspath('./data/processed_test.pkl'))

    parser.add_argument('--wo_vocab', type=str, default=os.path.abspath('./vocab/word_vocab.pkl'))
    parser.add_argument('--not_glove', action='store_false', help='do not use GloVe', dest='glove')
    parser.add_argument('--glove_path', type=str, default=os.path.abspath(''))
    parser.add_argument('--random_init_words', type=str,
                        default=os.path.abspath('./vocab/random_init_words.json'))
    parser.add_argument('--emb_cache', type=str, default=os.path.abspath('./vocab/word_embeddings_cache.pt'))
    parser.add_argument('--cpt', type=str, default='')

    parser.add_argument('--kb_endpoint', type=str, default='')

    args = parser.parse_args()
    return args
