#!/bin/bash

# task_types = ["ed", "ec", "re", "rc"]
# balance_variant = ['none', 'kcn', 'kt', 'nod', 'emp',
# 'sepcls_all_prev', 'sepcls_individual', 'replay', 'kd', 'ffe']

# - The correspondence between arguments and approaches are: 
#     - ICE: `sepcls_individual`
#     - ICE-PL: `sepcls_all_prev`
#     - Experience replay: `replay`
#     - Knowledge distillation: `kd`
#     - Finetune Feature Extractor variant: `ffe`
#     - [KCN](https://aclanthology.org/2020.emnlp-main.52/): `kcn`
#     - [Knowledge Transfer](https://aclanthology.org/2021.emnlp-main.428/): `kt`
#     - [Episodic Memory Prompt](https://aclanthology.org/2022.coling-1.189/): `emp`
# Since CRL requires much adaptation we will release its code in the future.

perm_id=(0)                           # decide which permutation of learning sessions to use
model_name="test_debug"
dataset="MAVEN"               
balance_variant="sepcls_individual"     # different balance algorithm
task="ed"                             # ed: event/entity detection; ec: event/entity classification; re: relation extraction; rc: relation classification
ec_train_other=0                      # use other instance to train or not
replay_flag_bool=0                    # 0 means do not apply experience replay
period=8                              # knowledge distillation / experience replay frequency
train_epoch=15
                                                    
gpu=0

if [ "$dataset" == "MAVEN" ]; then
    num_classes=169
elif [ "$dataset" == "ACE" ]; then
    num_classes=34
elif [ "$dataset" == "FewNERD" ]; then
    num_classes=67
elif [ "$dataset" == "TACRED" ]; then
    num_classes=41
fi

if [ "$task" == "ec" ] || [ "$task" == "rc" ]; then
    lr=0.0005
else
    lr=0.0001
fi

for tid in "${perm_id[@]}"
do
    python run_train.py \
      --dataset=$dataset \
      --feature-root="./data/features/${dataset}/" \
      --stream-file="data/${dataset}/streams.json" \
      --max-slots=$num_classes \
      --prompt-size=$num_classes \
      --perm-id=$tid \
      --ec-train-other=$ec_train_other \
      --save-model="${model_name}_$tid" \
      --load-model="${model_name}_$tid" \
      --eval-model-name=$model_name \
      --log-dir="logs/${model_name}_$tid" \
      --balance=$balance_variant \
      --learning-rate=$lr \
      --gpu=$gpu \
      --device="cuda:$gpu" \
      --task-type=$task \
      --period=$period \
      --replay-flag-bool=$replay_flag_bool \
      --train-epoch=$train_epoch \

done