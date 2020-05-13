# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : train.py
# @Software: PyCharm
"""

import os
import re
import sys
import time
import json
import glob
import math
import torch
import pickle
import random

sys.path.append("..")
import pargs
from data_loaders import RankingDataLoader
from models.ranking_model import RankingModel
from utils.query_interface import query_answers
from utils.utils import cal_score, check_query_equal


if __name__ == '__main__':
    args = pargs.ranking_pargs()

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('\nNote: You are using GPU for evaluation.\n')
        torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available() and not args.cuda:
        print('\nWarning: You have Cuda but do not use it. You are using CPU for evaluation.\n')

    wo_vocab = pickle.load(open(args.wo_vocab, 'rb'))
    print("load word vocab, size: %d" % len(wo_vocab))

    test_datas = pickle.load(open(args.test_data, "rb"))

    test_loader = RankingDataLoader(args)
    test_loader.load_data(test_datas, bs=1, use_small=args.use_small, shuffle=False)
    print("valid data, batch size: %d, batch number: %d" % (1, test_loader.n_batch))

    model = RankingModel(wo_vocab, args)
    if args.cuda:
        model.cuda()
        print('Shift model to GPU.\n')
    model.load_state_dict(torch.load(args.cpt))
    print("Load checkpoint from \"%s\"." % os.path.abspath(args.cpt))

    model.eval()

    avg_p, avg_r, avg_f1 = 0, 0, 0
    for s in test_loader.next_batch():
        data = s[-1][0]

        scores = model.ranking(s[:-1])
        pred_id = torch.argmax(scores).item()

        pred_processed_query = data["processed_cand_queries"][pred_id]
        gold_processed_query = data["processed_gold_query"]

        pred_query = data["cand_queries"][pred_id]
        gold_query = data["query"]

        pred_is_count = "COUNT" in pred_query
        pred_is_ask = "ASK" in pred_query

        gold_is_count = "COUNT" in gold_query
        gold_is_ask = "ASK" in gold_query

        if data["id"] in [4084, 4112, 4418, 4459, 4553, 4620, 4738, 4822, 4983, 4991,
                          4542, 4610,
                          4343]:
            continue

        if pred_is_ask == gold_is_ask and pred_is_count == gold_is_count \
                and check_query_equal(pred_processed_query, gold_processed_query):
            p, r, f1 = 1.0, 1.0, 1.0
        else:
            pred_answers = query_answers(pred_query, args.kb_endpoint)
            gold_answers = query_answers(gold_query, args.kb_endpoint)
            p, r, f1 = cal_score(pred_answers, gold_answers)

        avg_p += p
        avg_r += r
        avg_f1 += f1

    avg_p = avg_p * 100. / 1000
    avg_r = avg_r * 100. / 1000
    avg_f1 = avg_f1 * 100. / 1000

    print("\nAverage Precision: %.2f" % avg_p)
    print("Average Recall: %.2f" % avg_r)
    print("Average F1-score: %.2f\n" % avg_f1)
