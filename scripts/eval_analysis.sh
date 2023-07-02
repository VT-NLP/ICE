#!/bin/bash

# cd ..
# temp1="maven_ec_sepcls_0"
# temp2="maven_ec_sepcls_all_logit_inf_0"
# eval_model_name="maven_ec_bert_drifted_0"
# 'sepcls_all_prev', 'sepcls_individual', 'replay', 'kd', 'ffe']

task_id=(0)
#eval_model_name_list=("maven_ec_sepcls_0" "maven_ec_sepcls_all_logit_inf_0" "maven_ec_sepcls_indiv_train_other" "maven_ec_sepcls_all_logit_train_other")
eval_model_name_list=("allperm_maven_ec_sepcls_individual_0" "allperm_maven_ec_sepcls_individual_wother_0" )
balance_variant="sepcls"
task="ec"
gpu=2

for eval_model_name in "${eval_model_name_list[@]}"
do
    for tid in "${task_id[@]}"
    do
        python evaluate.py \
          --eval-loader-id=$tid \
          --model-dir="logs/$eval_model_name/$eval_model_name.$tid" \
          --balance=$balance_variant \
          --gpu=$gpu \
          --device="cuda:$gpu" \
          --eval-model-name=$eval_model_name \
          --task-type=$task \
          --case-study=1 \

    done
done
