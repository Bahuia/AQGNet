# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : eval.py
# @Software: PyCharm
"""

import os
import sys
import time
import json
import glob
import torch
import pickle

sys.path.append("..")
import pargs
from rules.grammar import AbstractQueryGraph
from data_loaders import GenerationDataLoader
from models.model import AQGNet
from utils.utils import kb_constraint, formalize_aqg, generate_cand_queries


if __name__ == '__main__':
    args = pargs.pargs()

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('\nNote: You are using GPU for evaluation.\n')
        torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available() and not args.cuda:
        print('\nWarning: You have Cuda but do not use it. You are using CPU for evaluation.\n')

    wo_vocab = pickle.load(open(args.wo_vocab, 'rb'))
    print("Load word vocab, size: %d" % len(wo_vocab))

    test_data = pickle.load(open(args.test_data, "rb"))

    test_loader = GenerationDataLoader(args)
    test_loader.load_data(test_data, bs=1, use_small=args.use_small, shuffle=False)
    print("Load valid data from \"%s\"." % (args.test_data))
    print("Test data, batch size: %d, batch number: %d" % (1, test_loader.n_batch))

    model = AQGNet(wo_vocab, args)
    if args.cuda:
        model.cuda()
        print('Shift model to GPU.\n')
    model.load_state_dict(torch.load(args.cpt))
    print("Load checkpoint from \"%s\"." % os.path.abspath(args.cpt))

    query_list = []
    n_q_correct, n_q_total = 0, 0
    model.eval()
    for s in test_loader.next_batch():
        data = s[-1][0]

        pred_aqg, action_probs = model.generation(s[:-1])

        pred_aqg = formalize_aqg(pred_aqg, data)

        if args.kb_constraint:
            pred_aqg = kb_constraint(aqg, data)

        is_correct = pred_aqg.is_equal(data["gold_aqg"])

        cand_queries = generate_cand_queries(pred_aqg, data)

        query_res = {
            "id": data["id"],
            "question": data["question"],
            "query": data["query"],
            "cand_queries": cand_queries
        }
        query_list.append(query_res)

        n_q_correct += is_correct
        n_q_total += 1

    acc = n_q_correct * 100. / n_q_total
    print("\nTotal AQG Accuracy: %.2f" % acc)

    checkpoint_dir = '/'.join(args.cpt.split('/')[:-2])

    results_path = os.path.join(checkpoint_dir, 'candidate_queries.json')
    json.dump(query_list, open(results_path, "w"), indent=4)
    print("Results save to \"{}\"\n".format(results_path))

