#!/bin/bash

devices=$1
save_name=$2
dbpedia_endpoint=$3     # http://10.201.158.104:3030/dbpedia/sparql

python eval.py \
--test_data ./data/processed_test.pkl \
--d_h 256 \
--gpu $devices \
--n_lstm_layers 1 \
--n_gnn_blocks 3 \
--heads 4 \
--beam_size 5 \
--cpt ${save_name} \
--kb_endpoint ${dbpedia_endpoint}