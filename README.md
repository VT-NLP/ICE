# Individual Classifier with Frozen Feature Extractor (ICE 🧊)

### Introduction 🚀
This is the official repository for the paper "[Teamwork Is Not Always Good: An Empirical Study of Classifier Drift in Class-incremental Information Extraction](https://arxiv.org/abs/2305.16559)" (Findings of ACL'23). We highlight the contributions as follows:

- We study a fundamental challenge, i.e., ***classifier drift***, in class-incremental learning.
- We introduce ICE 🧊, a super efficient and effective solution to classifier drift and catastrophic forgetting.
- For the first time, we conduct an extensive study and evaluation of class-incremental approaches on 6 essential IE tasks across 4 widely-used IE benchmarks.

### Basic Requirements 🔧
- Please make sure you have installed the following packages in your environment:
```
transformers==4.18.0
torch==1.7.1
torchmeta==1.8.0
numpy==1.19.5
tqdm==4.62.3
```
- You can install the requirements via running:
```
pip install -r requirements.txt
```

### Data Preparation 💾
- We conduct our study on 3 types of information extraction tasks across 4 popular benchmarks: 
    - **Named Entity Recognition** (FewNERD) 
    - **Relation Extraction** (TACRED) 
    - **Event Detection** (MAVEN, ACE) 
    <br><br>

    For each type of task, we consider two settings:
    - **Classification** that does not consider the *Other* class.
    - **Detection** that involves the *Other* class.
    <br><br>
    
    Please note that ACE is not publicly released and requires a license to access.
- First download the dataset files under the following directory with specified file names:
```
./data/{DATASET_NAME}/{DATASET_SPLIT}.jsonl
```
- Here `DATASET_NAME = {MAVEN, ACE, FewNERD, TACRED}, DATASET_SPLIT = {train, dev, test}`. Please make sure you have downloaded the files on all three splits. Also note that you need to preprocess the ACE dataset into the same format as MAVEN.
- Then run the follow script to compute and store the contextualized BERT features for each dataset:
```
python gen_pretrain_feature.py
```
The script will generate preprocessed files under the corresponding dataset directory. You can change the variable `dataset` inside to generate features for different datasets.

### Training & Evaluation ⚙️
- First create a directory`./logs/` which will stored the model checkpoints, and `./log/` which will stored log files. 
- Run the following script to start training. The script will also periodically evaluate the model on dev and test set.
```
./scripts/run_main.sh
```

Please see the comment in the script for more details on the argument. 


### Reference 📚
**Please consider citing our paper if find it useful or interesting.**
```
@inproceedings{liu-etal-2023-teamwork,
    title = "Teamwork Is Not Always Good: An Empirical Study of Classifier Drift in Class-incremental Information Extraction",
    author = "Liu, Minqian  and
      Huang, Lifu",
    booktitle = "Findings of the 61st Annual Meeting of the Association for Computational Linguistics",
    month = july,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics"
}
```

### Acknowledgement
Parts of the code in this repository are adopted from the work [Incremental Prompting](https://github.com/VT-NLP/Incremental_Prompting). We thank the members in VT NLP Lab for the constructive comments to this work.
