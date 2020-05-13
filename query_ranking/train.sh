#!/bin/bash

devices=$1
dbpedia_endpoint=$2

python train.py \
--train_data ../data/ranking_processed_train.pkl \
--valid_data ../data/ranking_processed_valid.pkl \
--glove_path ../data/GloVe/glove.42B.300d.txt \
--d_h_wo 512 \
--d_emb_wo 300 \
--gpu $devices \
--n_lstm_layers 1 \
--n_epochs 30 \
--ns 50 \
--bs 16 \
--lr 1e-3 \
--margin 0.1 \
--kb_endpoint ${dbpedia_endpoint}