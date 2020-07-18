#!/bin/bash

devices=$1

python train.py \
--train_data ./data/processed_train.pkl \
--valid_data ./data/processed_valid.pkl \
--glove_path ./data/glove.42B.300d.txt \
--readout identity \
--att_type affine \
--d_h 256 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--n_epochs 25 \
--bs 16 \
--lr 2e-4 \
--beam_size 7