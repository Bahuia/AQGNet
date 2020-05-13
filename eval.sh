#!/bin/bash

devices=0
save_name=/home/test2/yongrui.chen/AQGNet_local/runs/1589126395/checkpoints/best_snapshot_epoch_24_best_val_acc_79.6_model.pt
dbpedia_endpoint=http://10.201.180.179:3030/dbpedia/sparql

python eval.py \
--test_data ./data/processed_test.pkl \
--d_h 256 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--cpt ${save_name} \
--kb_endpoint ${dbpedia_endpoint}