# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/5/1
# @Author  : Yongrui Chen
# @File    : train.py
# @Software: PyCharm
"""

import os
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
from rules.grammar import AbstractQueryGraph
from data_loaders import GenerationDataLoader
from models.model import AQGNet


if __name__ == '__main__':
    args = pargs.pargs()

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('\nNote: You are using GPU for training.\n')
        torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available() and not args.cuda:
        print('\nWarning: You have Cuda but do not use it. You are using CPU for training.\n')

    wo_vocab = pickle.load(open(args.wo_vocab, 'rb'))
    print("load word vocab, size: %d" % len(wo_vocab))

    train_datas = pickle.load(open(args.train_data, "rb"))
    if args.shuffle:
        random.shuffle(train_datas)

    train_loader = GenerationDataLoader(args)
    train_loader.load_data(train_datas, args.bs, use_small=args.use_small, shuffle=args.shuffle)
    print("Load training data from \"%s\"."% (args.train_data))
    print("training data, batch size: %d, batch number: %d" % (args.bs, train_loader.n_batch))

    valid_datas = pickle.load(open(args.valid_data, "rb"))
    valid_loader = GenerationDataLoader(args)
    valid_loader.load_data(valid_datas, bs=1, use_small=args.use_small, shuffle=False)
    print("Load valid data from \"%s\"." % (args.valid_data))
    print("valid data, batch size: %d, batch number: %d" % (1, valid_loader.n_batch))

    model = AQGNet(wo_vocab, args)
    if args.cuda:
        model.cuda()
        print('Shift model to GPU.\n')

    # load pretrain embeddings.
    if args.glove:
        print('Loading pretrained word vectors from \"%s\" ...' % (args.glove_path))
        if os.path.isfile(args.emb_cache):
            pretrained_emb = torch.load(args.emb_cache)
            model.word_embedding.word_lookup_table.weight.data.copy_(pretrained_emb)
        else:
            pretrained_emb, random_init_words = model.word_embedding.load_pretrained_vectors(
                args.glove_path, binary=False, normalize=args.word_normalize)
            json.dump(random_init_words, open(args.random_init_words, 'w', encoding='utf-8'), indent=2)
            torch.save(pretrained_emb, args.emb_cache)

    # optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # create runs directory.
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('\nModel writing to \"{}\"\n'.format(out_dir))
    with open(os.path.join(out_dir, 'param.log'), 'w') as fin:
        fin.write(str(args))
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for name, value in model.named_parameters():
        print(name, value.size())

    iters = 0
    start = time.time()
    best_val_q_acc = 0
    best_lv4_q_acc = 0

    header = '\n  Time Epoch         Loss    Train_Step_Acc        Train_Acc        Valid_Acc'
    val_log_template = ' '.join(
        '{:>6.0f},{:>5.0f},{:>12.6f},{:16.4f},{:16.4f},{:16.4f}'.split(','))
    best_snapshot_prefix = os.path.join(checkpoint_dir, 'best_snapshot')

    print('\nTraining start.')

    print(header)

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        avg_loss = 0.
        n_q_total = 0
        for i, b in enumerate(train_loader.next_batch()):
            data = b[-1]
            optimizer.zero_grad()

            loss, _ = model(b[:-1])

            loss = torch.mean(loss)
            loss.backward()
            # clip the gradient.
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            avg_loss += loss.data.cpu().numpy()*len(data)
            n_q_total += len(data)
        avg_loss /= n_q_total

        model.eval()
        n_q_correct, n_step_correct, n_step_total = 0, 0, 0
        for i, b in enumerate(train_loader.next_batch()):
            gold_objs = b[-2]
            _, action_probs = model(b[:-1])
            for s_id in range(len(gold_objs)):
                is_correct = True
                for j in range(len(gold_objs[s_id])):
                    pred_obj = torch.argmax(action_probs[s_id][j], dim=-1).item()
                    if pred_obj == gold_objs[s_id][j]:
                        n_step_correct += 1
                    else:
                        is_correct = False
                n_step_total += len(gold_objs[s_id])
                n_q_correct += is_correct

        train_q_acc = 100. * n_q_correct / n_q_total
        train_step_acc = 100. * n_step_correct / n_step_total

        val_n_q_correct, val_n_q_total = 0, 0
        for s in valid_loader.next_batch():
            data = s[-1][0]
            pred_aqg, action_probs = model.generation(s[:-1])

            if pred_aqg.is_equal(data["gold_aqg"]):
                val_n_q_correct += 1
            val_n_q_total += 1

        val_q_acc = val_n_q_correct * 100. / val_n_q_total

        print(val_log_template.format(time.time() - start, epoch, avg_loss,
                                      train_step_acc, train_q_acc, val_q_acc))

        # update checkpoint.
        if val_q_acc >= best_val_q_acc:
            best_val_q_acc = val_q_acc
            snapshot_path = best_snapshot_prefix + \
                            '_epoch_{}_best_val_acc_{}_model.pt'.format(epoch, best_val_q_acc)
            # save model, delete previous 'best_snapshot' files.
            torch.save(model.state_dict(), snapshot_path)
            for f in glob.glob(best_snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

    print('\nTraining finished.')
    print("\nBest Acc: {:.2f}\nModel writing to \"{}\"\n".format(best_val_q_acc, out_dir))