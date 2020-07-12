#!/bin/bash

train_data=./data/processed_train.pkl
valid_data=./data/processed_valid.pkl
test_data=$1            # structure prediction results path
dbpedia_endpoint=$2     # http://10.201.158.104:3030/dbpedia/sparql


echo "Start generate candidate queries for training data ..."
python generate_queries.py \
--data_path ${train_data} \
--use_gold_structure \
--kb_endpoint ${dbpedia_endpoint} \
--output ./data/train_candidate_queries/

echo "Start generate candidate queries for valid data ..."
python generate_queries.py \
--data_path ${valid_data} \
--use_gold_structure \
--kb_endpoint ${dbpedia_endpoint} \
--output ./data/valid_candidate_queries/

echo "Start generate candidate queries for test data ..."
python generate_queries.py \
--data_path ${test_data} \
--kb_endpoint ${dbpedia_endpoint} \
--output ./data/test_candidate_queries/