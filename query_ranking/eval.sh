#!/bin/bash

devices=$1
save_name=$2
dbpedia_endpoint=$3

python eval.py \
--test_data ../data/ranking_processed_test.pkl \
--d_h_wo 512 \
--d_emb_wo 300 \
--gpu $devices \
--cpt ${save_name} \
--kb_endpoint ${dbpedia_endpoint}