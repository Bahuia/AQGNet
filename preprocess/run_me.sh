#!/bin/bash

train=../data/train.json
valid=../data/valid.json
test=../data/test.json

output_train=../data/processed_train.pkl
output_valid=../data/processed_valid.pkl
output_test=../data/processed_test.pkl

echo "Start download NLTK data ..."
python download_nltk.py
echo "Finish.\n"


echo "Start process the train data ..."

python -u preprocess.py \
--data_path ${train} \
--output ${output_train}

echo "Finish.\n"

echo "Start process the valid data ..."

python -u preprocess.py \
--data_path ${valid} \
--output ${output_valid}

echo "Finish.\n"

echo "Start process the test data ..."

python -u preprocess.py \
--data_path ${test} \
--output ${output_test}

echo "Finish.\n"