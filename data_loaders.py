# -*- coding: utf-8 -*-
# !/usr/bin/python
import sys
import math
import torch
import copy
import json
import random
import pickle
import numpy as np

sys.path.append("..")
from pargs import pargs
from utils.utils import pad, mk_graph_for_gnn
from utils.utils import text_to_tensor_1d, text_to_tensor_2d, pad_tensor_1d, pad_tensor_2d
from rules.grammar import AbstractQueryGraph


class GenerationDataLoader:

    def __init__(self, args):
        self.args = args
        self.wo_vocab = pickle.load(open(self.args.wo_vocab, 'rb'))

        self.wo_pad = self.wo_vocab.lookup(self.wo_vocab.pad_token)

    def process_one_data(self, d):

        question_toks = d["question_toks"]
        q = text_to_tensor_1d(question_toks, self.wo_vocab)
        gold_objs = d["rule_labels"]

        gold_graphs = []
        g = AbstractQueryGraph()
        g.init_state()
        for i, obj in enumerate(gold_objs):
            vertices, v_labels, edges = g.get_state()
            gold_graphs.append(mk_graph_for_gnn(vertices, v_labels, edges))
            op = g.cur_operation
            g.update_state(op, obj)

        return q, gold_graphs, gold_objs, d


    def load_data(self, file, bs, use_small=False, shuffle=True):
        datas = pickle.load(open(file, "rb"))

        if use_small:
            datas = datas[:100]

        if shuffle:
            random.shuffle(datas)

        bl_x = []
        batch_index = -1  # the index of sequence batches
        sample_index = 0  # sequence index within each batch

        for d in datas:
            if sample_index % bs == 0:
                sample_index = 0
                batch_index += 1
                bl_x.append([])
            x = self.process_one_data(d)
            bl_x[batch_index].append(x)
            sample_index += 1

        self.iters = []
        self.n_batch = len(bl_x)
        for x in bl_x:
            batch = self.fix_batch(x)
            self.iters.append(batch)

    def fix_batch(self, x):

        q, gold_graphs, gold_objs, data = zip(*x)
        q, q_lens = pad_tensor_1d(q, self.wo_pad)

        if self.args.cuda:
            q = q.to(self.args.gpu)
            gold_graphs = [[[y.to(self.args.gpu) for y in g] for g in s] for s in gold_graphs]

        return q, q_lens, gold_graphs, gold_objs, data

    def next_batch(self):
        for b in self.iters:
            yield b