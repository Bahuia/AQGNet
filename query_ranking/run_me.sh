#!/bin/bash


train_data=../data/train_candidate_queries
valid_data=../data/valid_candidate_queries
test_data=../data/test_candidate_queries_1590760970

output_train=../data/ranking_processed_train.pkl
output_valid=../data/ranking_processed_valid.pkl
output_test=../data/ranking_processed_test.pkl


echo "Start process ranking training data ..."
python -u preprocess.py \
--data_path ${train_data} \
--output ${output_train}
echo "Finish.\n"

echo "Start process ranking valid data ..."
python -u preprocess.py \
--data_path ${valid_data} \
--output ${output_valid}
echo "Finish.\n"

echo "Start process ranking test data ..."
python -u preprocess.py \
--data_path ${test_data} \
--output ${output_test}
echo "Finish.\n"