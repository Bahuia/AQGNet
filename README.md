### Important Notice
Our latest work [*Outlining and Filling: Hierarchical Query Graph Generation for Answering Complex Questions over Knowledge Graphs*](https://arxiv.org/abs/2111.00732) has been accepted by IEEE Transactions on Knowledge and Data Engineering. All [source codes](https://github.com/Bahuia/HGNet) are available for research purposes. 

### Overview
Data and code for IJCAI 2020 paper [Formal Query Building with Query Structure Prediction for Complex Question Answering over Knowledge Base](https://www.ijcai.org/Proceedings/2020/0519.pdf) is available for research purposes.  

This project only includes the processing of the LC-QuAD [(Trivedi et al., 2017)](http://jens-lehmann.org/files/2017/iswc_lcquad.pdf) dataset, and we are sorry that the source code of the remaining two data sets, WebQuestions [(Berant et al., 2013)](https://www.aclweb.org/anthology/D13-1160.pdf) and ComplexQuestions [(Bao et al, 2016)](https://www.aclweb.org/anthology/C16-1236.pdf), cannot be released due to the lack of organization currently. We will release them in a unified way in future work.  In view of the preprocessing data link provided by [(Luo et al., 2018)](https://www.aclweb.org/anthology/D18-1242.pdf) for WebQ and CompQ is no longer valid, we provide a new link [here](https://drive.google.com/file/d/1bL5vEIek9kBDe_IKbicl5y_6k-fQ6Ogd/view?usp=sharing) for subsequent researchers. 

### Requirements
* Python 3.6
* Pytorch 1.2.0
* DBpedia Version [2016-04](https://drive.google.com/file/d/17TRlj8a34IEo686nnKHTewZEg4aMrYUe/view?usp=sharing) (Note the version. If you use the latest DBpedia version, the answers to some questions will not be retrieved. Here, we also performed a preprocessing on it, and only retained the English part related to the LC-QuAD data set.)
* SPARQL service (constructed by Virtuoso or Apache Jena Fuseki)

### Update
Recently, we have updated AQGNet, we changed AQG from **undirected graph** to the **directed graph** and added *beam search* in structure prediction. According to this update, the performance of our approach has been further improved on the LC-QuAD.

| **Dataset**   | AQG prediction | Precision | Recall | F1-score |
| ------------- | :------------: | :-------: | :----: | :------: |
|   LC-QuAD     |  72.8          | 77.38     | 76.73  | 76.59    |

#### Download Trained Model
We provide the above [trained AQGNet model](https://drive.google.com/file/d/1PPs2u4CX_qraFhl1YNQTbvp-MAvhKRbd/view?usp=sharing). Please download and unzip, move it to the `./runs` directory.

Here we provide the [candidate queries](https://drive.google.com/file/d/10fKDMPHxO-w85TWuCwHS1zW9tv9zHlOs/view?usp=sharing) of the training set, the verification set and the test set respectively, where the candidate queries of the test set are obtained from the prediction results of the above model. Please download and unzip, move it to the `./data` directory.

We also provide the [trained query ranking model](https://drive.google.com/file/d/1p1JnQlTQ2kA-ZnRoAKYOqu7FWwzX-dT8/view?usp=sharing). Please download and unzip, move it to the `./query_ranking/runs` directory.

### Running Code
Download [Glove Embedding](http://nlp.stanford.edu/data/glove.42B.300d.zip) and put `glove.42B.300d.txt` under `./data/` directory.

#### 1. Preprocess data for AQGNet
```bash
cd ./preprocess
sh run_me.sh
```


#### 2. Training for AQGNet
Modify the following content in `./train.sh`.
```bash
devices=$1
```
* Replace `$1` with the id of the GPU to be used, such as `0`.  
Then, execute the following command for training.
```bash
sh train.sh
```
The trained model file is saved under `./runs` directory.  
The path format of the trained model is `./runs/RUN_ID/checkpoints/best_snapshot_epoch_xx_best_val_acc_xx_model.pt`


#### 3. Testing for AQGNet
Modify the following content in `./eval.sh`.
```bash
devices=$1
save_name=$2
dbpedia_endpoint=$3
```
* Replace `$1` with the id of the GPU to be used.  
* Replace `$2` with the path of the trained model.  
* Replace `$3` with the address of the established DBpedia SPARQL service, such as `http://10.201.158.104:3030/dbpedia/sparql`

The result of AQGNet structure prediction is saved under the used model directory. The path format of result is `./runs/RUN_ID/results.pkl`.  
Then, execute the following command for structure prediction.
```bash
sh eval.sh
```


#### 4. Generate candidate queries
Modify the following content in `./generate_queries.sh`.
```bash
test_data=$1            # structure prediction results path
dbpedia_endpoint=$2     # http://10.201.158.104:3030/dbpedia/sparql
```
The candidate queries for the training set, valid set, and test set are saved under `./data` directory.


#### 5. Preprocess data for query ranking model
```bash
cd ./query_ranking
sh run_me.sh
```

#### 6. Training for query ranking model
Modify the following content in `./query_ranking/train.sh`.
```bash
devices=$1
```
* Replace `$1` with the id of the GPU to be used.
Then, execute the following command for training query ranking model.
```bash
cd ./query_ranking
sh train.sh
```
The trained query ranking model file is saved under `./query_ranking/runs` directory. 

#### 6. Test for query ranking model
Modify the following content in `./query_ranking/eval.sh`.
```bash
devices=$1
save_name=$2
dbpedia_endpoint=$3
```
* Replace `$1` with the id of the GPU to be used.  
* Replace `$2` with the path of the trained model.  
* Replace `$3` with the address of the established DBpedia SPARQL service, such as `http://10.201.158.104:3030/dbpedia/sparql`.

Then, execute the following command for the final results of question answering.
```bash
cd ./query_ranking
sh eval.sh
```

### Citation
If you use AQGNet, please cite the following work.
```
@inproceedings{DBLP:conf/ijcai/ChenLHQ20,
  author    = {Yongrui Chen and
               Huiying Li and
               Yuncheng Hua and
               Guilin Qi},
  editor    = {Christian Bessiere},
  title     = {Formal Query Building with Query Structure Prediction for Complex
               Question Answering over Knowledge Base},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2020 [scheduled for July 2020, Yokohama,
               Japan, postponed due to the Corona pandemic]},
  pages     = {3751--3758},
  publisher = {ijcai.org},
  year      = {2020},
  url       = {https://doi.org/10.24963/ijcai.2020/519},
  doi       = {10.24963/ijcai.2020/519},
  timestamp = {Mon, 13 Jul 2020 18:09:15 +0200},
  biburl    = {https://dblp.org/rec/conf/ijcai/ChenLHQ20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
