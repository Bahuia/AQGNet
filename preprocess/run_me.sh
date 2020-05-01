#!/bin/bash

train=../data/train.json
test=../data/test.json
output1=../data/processed_train.pkl
output2=../data/processed_test.pkl

echo "Start download NLTK data ..."
python download_nltk.py
echo "Finish.\n"


echo "Start process the origin LC-QuAD training data ..."

python -u preprocess.py \
--data_path ${train} \
--output ${output1}
echo "Finish.\n"

echo "Start process the origin LC-QuAD test data ..."

python -u preprocess.py \
--data_path ${test} \
--output ${output2}

echo "Finish.\n"