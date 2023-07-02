import pdb
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from tqdm import tqdm
from utils.optimizer import AdamW
from utils.options import parse_arguments
# from utils.datastream import get_stage_loaders, get_stage_loaders_n
import sys

# from utils.dataloader_ace import get_stage_loaders, get_stage_loaders_n

from utils.worker import Worker
from models.emp import PromptNet
# from models.adapter.prior_adapter import BERT
from utils.utils import get_task_stat

# from models.bert_baseline import BERT, BIC, ICARL
from models.prev_baseline import BERT, BIC, ICARL

from models.sep_cls import SepCLS
from models.sep_cls_ft import SepCLSFT

# from models.baseline import KDR,
import random
import csv
from contextlib import redirect_stdout

opts = parse_arguments()
test_only = True
print(f"test: {opts.eval_model_name}")
print(f"test: {opts.model_dir}")

if opts.task_type == 'ec' or opts.task_type == "rc":
    from utils.dataloader_no_other import get_stage_loaders, get_stage_loaders_n
else:
    from utils.dataloader import get_stage_loaders, get_stage_loaders_n

print(opts.save_model)
if not opts.ec_train_other:
    print("EC not train with other")
else:
    print("train with other")

# if opts.balance == "sepcls_ft":
#     lr = opts.learning_rate / 100
# else:
#     lr = opts.learning_rate

# print(f"Learning Rate: {lr}")
lr = opts.learning_rate
# PERM = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 0, 3, 1, 4], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

PERM, TASK_NUM, TASK_EVENT_NUM, NA_TASK_EVENT_NUM, ACC_NUM = get_task_stat(opts.dataset, opts.perm_id)


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def by_class(preds, labels, learned_labels=None):
    match = (preds == labels).float()
    nlabels = max(torch.max(labels).item(), torch.max(preds).item())
    bc = {}
    type_dist = {}
    ag = 0; ad = 0; am = 0
    for label in range(1, nlabels+1):
        lg = (labels==label); ld = (preds==label)
        type_dist[label] = int(torch.sum(lg.float()).item())
        lr = torch.sum(match[lg]) / torch.sum(lg.float())
        lp = torch.sum(match[ld]) / torch.sum(ld.float())
        lf = 2 * lr * lp / (lr + lp)
        if torch.isnan(lf):
            bc[label] = (0, 0, 0)
        else:
            bc[label] = (lp.item(), lr.item(), lf.item())
        if learned_labels is not None and label in learned_labels:
            ag += lg.float().sum()
            ad += ld.float().sum()
            am += match[lg].sum()
    if learned_labels is None:
        ag = (labels!=0); ad = (preds!=0)
        sum_ad = torch.sum(ag.float())
        if sum_ad == 0:
            ap = ar = 0
        else:
            ar = torch.sum(match[ag]) / torch.sum(ag.float())
            ap = torch.sum(match[ad]) / torch.sum(ad.float())
    else:
        if ad == 0:
            ap = ar = 0
        else:
            ar = am / ag; ap = am / ad
    if ap == 0:
        af = ap = ar = 0
    else:
        af = 2 * ar * ap / (ar + ap)
        af = af.item(); ar = ar.item(); ap = ap.item()
    # print(type_dist)
    return bc, (ap, ar, af)


def new_and_old(per_type_f1s):

    if TASK_NUM == 1:
        return
    new_f1s = []
    acc_old_f1s = [0]
    j = 0
    performance_list = []
    accumulate_old_type_f1_per_task = []
    for i in range(TASK_NUM):
        performance_list.append(list(per_type_f1s[i].values()))
    for t in range(TASK_NUM):
        curr_f1 = sum(performance_list[t][j:TASK_EVENT_NUM[t] + j]) / len(performance_list[t][j:TASK_EVENT_NUM[t] + j])
        new_f1s.append(curr_f1)
        j = TASK_EVENT_NUM[t] + j
        if t > 0:
            acc_old_f1 = sum(performance_list[t][0:ACC_NUM[t]]) / len(performance_list[t][0:ACC_NUM[t]])
            acc_old_f1s.append(acc_old_f1)

            k = 0
            old_type_f1_per_task = []
            for i in range(t):
                old_f1 = sum(performance_list[t][k:TASK_EVENT_NUM[i] + k]) / len(
                    performance_list[t][k:TASK_EVENT_NUM[i] + k])
                k = TASK_EVENT_NUM[i] + k
                old_type_f1_per_task.append(old_f1)
            old_type_f1_per_task = [round(100 * i, 2) for i in old_type_f1_per_task]
            accumulate_old_type_f1_per_task.append(old_type_f1_per_task)



    new_f1s = [round(100 * i, 2) for i in new_f1s]
    acc_old_f1s = [round(100 * i, 2) for i in acc_old_f1s]

    print("New Type F1:")
    print(new_f1s)
    print("Accumulate Old Type F1:")
    print(acc_old_f1s)
    print("Per Task Old Type F1:")
    print(accumulate_old_type_f1_per_task)

    return new_f1s, acc_old_f1s, accumulate_old_type_f1_per_task


def main():
    
    opts = parse_arguments()
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    summary = SummaryWriter(opts.log_dir)

    if test_only:
        loader_id = opts.eval_loader_id

    dataset_id = 0
    # if 'MAVEN' in opts.log_dir:
    #     dataset_id = 0
    # elif 'ACE' in opts.log_dir:
    #     dataset_id = 1

    if opts.balance == "kt":
        opts.kt = True

    perm_id = opts.perm_id
    if opts.setting == "classic":
        streams = json.load(open(opts.stream_file))
        streams = [streams[t] for t in PERM[perm_id]]
        loaders, dev_loaders, test_loaders, exemplar_loaders, stage_labels, label2id = get_stage_loaders(root=opts.json_root,
            feature_root=opts.feature_root,
            batch_size=opts.batch_size,
            streams=streams,
            num_workers=1,
            dataset=dataset_id)
    else:
        sis = json.load(open("data/MAVEN/stream_instances_2227341903.json"))
        if perm_id <= 3:
            print(f"running perm {perm_id}")
            sis = [sis[t] for t in PERM[perm_id]]
        loaders, exemplar_loaders, stage_labels, label2id = get_stage_loaders_n(root=opts.json_root,
            feature_root=opts.feature_root,
            batch_size=opts.batch_size,
            streams=json.load(open(opts.stream_file)),
            streams_instances=sis,
            num_workers=1,
            dataset=dataset_id)
    if opts.balance == 'bic':
        model = BIC(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0])+1 if not test_only else ACC_NUM[loader_id+1],
            device=torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
        )
    elif opts.balance == "icarl":
        model = ICARL(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0])+1 if not test_only else max(stage_labels[-1])+1,
            device=torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
        )
    elif opts.balance == "emp":
        model = PromptNet(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0]) + 1 if not test_only else ACC_NUM[loader_id+1],
            label_mapping=label2id,
            device=torch.device(
                torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu')),
            task_id=loader_id
        )
    elif opts.balance == "sepcls_all_prev" or opts.balance == "sepcls_individual" or opts.balance == "sepcls":
        model = SepCLS(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0]) + 1 if not test_only else ACC_NUM[loader_id+1],
            label_mapping=label2id,
            device=torch.device(
                torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu')),
            task_id=loader_id

        )
    else:
        model = BERT(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0])+1 if not test_only else ACC_NUM[loader_id+1],
            label_mapping=label2id,
            device=torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu')),
            task_id=loader_id
        )
    param_groups = [
        {"params": [param for name, param in model.named_parameters() if param.requires_grad and 'correction' not in name],
        "lr":opts.learning_rate,
        "weight_decay": opts.decay,
        "betas": (0.9, 0.999)}
        ]
    optimizer = AdamW(params=param_groups)
    optimizer_correction = None
    if opts.balance == "bic":
        correction_param_groups = [
            {"params": [param for name, param in model.named_parameters() if param.requires_grad and "correction_weight" in name],
            "lr":opts.learning_rate,
            "weight_decay": 0,
            "betas": (0.9, 0.999)},
            {"params": [param for name, param in model.named_parameters() if param.requires_grad and "correction_bias" in name],
            "lr":opts.learning_rate,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999)}
        ]
        assert len(correction_param_groups[0]['params']) == 1
        assert len(correction_param_groups[1]['params']) == 1
        optimizer_correction = AdamW(params=correction_param_groups)
    worker = Worker(opts)
    worker._log(str(opts))
    worker._log(str(label2id))
    id2label = {}
    for label, id in label2id.items():
        id2label[id] = label
    print(json.dumps(id2label))
    if test_only:
        worker.load(model, path=opts.model_dir)
    # ada_gating_dist = []
    # l = opts.num_hidden_layers
    # valid_idx = [i for i in range(0, l)] + [i for i in range(l * 2, l * 3)]
    # for idx, ada in enumerate(model.adapters):
    #     if idx in valid_idx:
    #         ada_gating_dist.append(ada.gating_dist)
    #
    # for i in range(len(ada_gating_dist)):
    #     for j in range(opts.num_expert):
    #         print('%.2f' % (ada_gating_dist[i][j] / 8038), end=" ")
    #     print()
    pdb.set_trace()
    best_dev = best_test = None
    collect_stats = "accuracy"
    collect_outputs = {"prediction", "label"}
    termination = False
    patience = opts.patience
    no_better = 0

    if test_only:
        loader_id = opts.eval_loader_id

    total_epoch = 0
    none_mul = 4
    learned_labels = set(stage_labels[0])
    best_dev_scores = []
    best_test_scores = []
    dev_metrics = None
    test_metrics = None
    exemplar_flag = opts.replay_flag
    # TODO: test set examplar
    # print("TEST: setting train exemplar for learned classes")
    # model.set_exemplar(exemplar_loaders[loader_id])
    # model.set_history()
    while not termination:
        if not test_only:
            if opts.skip_first and loader_id == 0:
                worker.load(model, optimizer, path=opts.load_first, strict=opts.balance!='bic')
                total_epoch += worker.epoch
            elif opts.skip_second and loader_id == 1:
                worker.load(model, optimizer, path=opts.load_second, strict=opts.balance!='bic')
                total_epoch += worker.epoch
            else:
                if opts.finetune:
                    train_loss = lambda batch:model.forward(batch)
                elif opts.balance == "bic" and loader_id >= 2:
                    train_loss = lambda batch:model.forward(batch, bias_correction="last", exemplar=True, exemplar_distill=True, distill=True, feature_distill=False,tau=0.5, task_id=loader_id)
                elif opts.balance == "kcn":
                    train_loss = lambda batch:model.forward(batch, exemplar=exemplar_flag, feature_distill=True, exemplar_distill=exemplar_flag, distill=True, tau=0.5, task_id=loader_id)
                elif opts.balance == "kt":
                    train_loss = lambda batch:model.forward(batch, exemplar=exemplar_flag, mul_distill=True, exemplar_distill=exemplar_flag, distill=True, tau=0.5, task_id=loader_id)
                elif opts.balance == "emp":
                    train_loss = lambda batch: model.forward(batch, exemplar=exemplar_flag, exemplar_distill=True,
                                                             distill=False, feature_distill=True, tau=0.5,
                                                             task_id=loader_id)
                else:
                    train_loss = lambda batch:model.forward(batch, exemplar=exemplar_flag, exemplar_distill=False, distill=False, feature_distill=False, tau=0.5, task_id=loader_id, train_mode=True)
                epoch_loss, epoch_metric = worker.run_one_epoch(
                    model=model,
                    f_loss=train_loss,
                    loader=loaders[loader_id],
                    split="train",
                    optimizer=optimizer,
                    collect_stats=collect_stats,
                    prog=loader_id)
                total_epoch += 1
                # reset iter counter
                model.iter_cnt = 0
                # shuffle examplar index
                if loader_id > 0 and exemplar_flag:
                    random.seed(opts.seed+99*total_epoch)
                    random.shuffle(model.random_exemplar_inx)

                for output_log in [print, worker._log]:
                    output_log(
                        f"Epoch {worker.epoch:3d}  Train Loss {epoch_loss} {epoch_metric}")
        else:
            learned_labels = set([t for stream in stage_labels for t in stream])
            termination = True

        if test_only:
            if opts.balance == "icarl":
                exemplar = model.set_exemplar(exemplar_loaders[loader_id], output_only=True)
                exemplar_features = []
                exemplar_labels = []
                for label, features in exemplar.items():
                    exemplar_features.append(features)
                    exemplar_labels.extend([label]*features.size(0))
                exemplar_features = torch.cat(exemplar_features, dim=0).cpu()
                exemplar_labels = torch.LongTensor(exemplar_labels).cpu()
                if model.exemplar_features is not None:
                    exemplar_features = torch.cat((model.exemplar_features, exemplar_features), dim=0)
                    exemplar_labels = torch.cat((model.exemplar_labels, exemplar_labels), dim=0)
                model.set_none_feat(loaders[loader_id])
                score_fn = lambda t:model.score(t, exemplar=(exemplar_labels, exemplar_features))
            else:
                # score_fn = model.score
                score_fn = lambda batch: model.forward(batch, exemplar=False, exemplar_distill=False, distill=False,
                                                       feature_distill=False, tau=0.5, task_id=loader_id,
                                                       train_mode=False)
            # dev_loss, dev_metrics = worker.run_one_epoch(
            #     model=model,
            #     f_loss=score_fn,
            #     loader=loaders[-2],
            #     split="dev",
            #     collect_stats=collect_stats,
            #     collect_outputs=collect_outputs)
            # dev_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
            # dev_scores, (dev_p, dev_r, dev_f) = by_class(dev_outputs["prediction"], dev_outputs["label"], learned_labels=learned_labels)
            # dev_class_f1 = {k: dev_scores[k][2] for k in dev_scores}
            # for k,v in dev_class_f1.items():
            #     add_summary_value(summary, f"dev_class_{k}", v, total_epoch)
            # dev_metrics = dev_f
            # for output_log in [print, worker._log]:
            #     output_log(
            #         f"Epoch {worker.epoch:3d}:  Dev {dev_metrics}"
            #     )
            if opts.task_type == 'ec':
                test_loader_tmp = test_loaders[loader_id]
            else:
                test_loader_tmp = test_loaders
            test_loss, test_metrics = worker.run_one_epoch(
                model=model,
                f_loss=score_fn,
                loader=test_loader_tmp,
                split="test",
                collect_stats=collect_stats,
                collect_outputs=collect_outputs)
            test_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
            torch.save(test_outputs, f"log/{os.path.basename(opts.load_model)}.output")
            test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"], learned_labels=learned_labels)
            test_class_f1 = {k: test_scores[k][2] for k in test_scores}
            print(model.task1_gold_logit_sum)
            print(model.task1_instance_cnt)
            print("************ Avg Gold Logit on Task 1 ************")
            print(model.task1_gold_logit_sum / model.task1_instance_cnt)
            print("************ Avg Pred Logit on Task 1 ************")
            print(model.curr_pred_logit_sum / model.task1_instance_cnt)

            avg_task_logit = model.mean_task_logit / model.task_instance_cnt
            avg_task_logit = avg_task_logit.tolist()
            avg_mean_task1_logit = model.mean_task1_logit / model.task1_instance_cnt
            avg_mean_task1_logit = avg_mean_task1_logit.tolist()

            avg_task_prob = model.mean_task_prob / model.task_instance_cnt
            avg_task_prob = avg_task_prob.tolist()
            avg_mean_task1_prob = model.mean_task1_prob / model.task1_instance_cnt
            avg_mean_task1_prob = avg_mean_task1_prob.tolist()

            # print("redirecting output...")
            # origin_stdout = sys.stdout
            # with open(f"./analysis/eval_output/{opts.eval_model_name}_{opts.eval_loader_id}.txt", "w") as af:
            #     sys.stdout = af
            #     print(f"Number of task instance: {task_instance_cnt}")
            #     print(f"Averaged probability: {avg_task_prob}")
            #     print(f"Averaged probability for Task 1: {avg_mean_task1_prob}")
            #     print(f"Accumulate Number of span count: {model.span_cnt}")
            #     sys.stdout = origin_stdout

            # with open(f'analysis/logit_analysis/{opts.eval_model_name}_logit.txt', 'a+') as f:
            #     with redirect_stdout(f):
            #         print(f"Session Number: {opts.eval_loader_id}")
            #         print("************ Avg Gold Logit on Task 1 ************")
            #         print(model.task1_gold_logit_sum / model.task1_instance_cnt)
            #         print("************ Avg New classifier's Pred Logit on Task 1 ************")
            #         print(model.curr_pred_logit_sum / model.task1_instance_cnt)


            logit_path =  f"./analysis/eval_output/{opts.eval_model_name}.json"
            dict_key = f"{opts.eval_model_name}-{opts.eval_loader_id}"
            overall_dict_logit = {dict_key+"-overall-logit": avg_task_logit}
            task1_dict_logit = {dict_key+"-task1-logit": avg_mean_task1_logit}

            overall_dict_prob = {dict_key + "-overall-prob": avg_task_prob}
            task1_dict_prob = {dict_key + "-task1-prob": avg_mean_task1_prob}

            with open(logit_path, "a+") as cf:
                # data = overall_dict
                # data.append(task1_dict)
                json.dump(overall_dict_logit, cf)
                cf.write('\n')
                json.dump(task1_dict_logit, cf)
                cf.write('\n')
                json.dump(overall_dict_prob, cf)
                cf.write('\n')
                json.dump(task1_dict_prob, cf)
                cf.write('\n')

            # ada_gating_dist = []
            # ada_gating_logit = []
            # l = opts.num_hidden_layers
            # valid_idx = [i for i in range(0, l)] + [i for i in range(l * 2, l * 3)]
            # for idx, ada in enumerate(model.adapters):
            #     if idx in valid_idx:
            #         ada_gating_dist.append(ada.gating_dist)
            #         ada_gating_logit.append(ada.gating_logit)
            #
            # print("-------gating dist-------")
            # for i in range(len(ada_gating_dist)):
            #     for j in range(opts.num_expert):
            #         print('%.2f' % (ada_gating_dist[i][j]/8038), end=", ")
            #     print()
            #
            # print("-------gating logit-------")
            # for i in range(len(ada_gating_logit)):
            #     for j in range(opts.num_expert):
            #         print('%.2f' % (ada_gating_logit[i][j]/8038), end=", ")
            #     print()

            # TODO: switch for case study


            case_out = model.analysis_out
            # save output example
            with open(f"./analysis/case_study/{opts.eval_model_name}_{opts.eval_loader_id}.csv", "w") as of:
                # pdb.set_trace()
                # json.dump(case_out, of)
                writer = csv.writer(of)
                # key_list = list(case_out.keys())
                # limit = len(key_list)
                writer.writerow(case_out.keys())
                writer.writerows(zip(*case_out.values()))

            print("------F1 per class------")

            print(json.dumps(test_class_f1))
            # test_outs = {"input_ids": test_outputs["input_ids"], "prediction": test_outputs["prediction"], "label": test_outputs["label"]}

            # with open('emp4_out.json', 'w', encoding='utf-8') as f:
            #     json.dump(test_outs, f, ensure_ascii=False)
            # with open('emp0_label.json', 'w', encoding='utf-8') as f:
            #     json.dump(test_outputs["label"], f, indent=4)

            for k,v in test_class_f1.items():
                add_summary_value(summary, f"test_class_{k}", v, total_epoch)
            test_metrics = test_f
            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}: Test {test_metrics}"
                )
            # if opts.test_only:
            #     frequency = {}
            #     for loader in loaders[:-2]:
            #         indices = loader.dataset.label2index
            #         for label in indices.keys():
            #             if label != 0:
            #                 frequency[label] = indices[label][1] - indices[label][0]
            #     with open("data/MAVEN/label2id.json") as fp:
            #         name2label = json.load(fp)
            #         label2name = {v:k for k,v in name2label.items()}
            #     id2label = {v:k for k,v in label2id.items()}
            #     sf = [(frequency[l], label2name[id2label[l]], dev_class_f1[l], test_class_f1[l]) for l in frequency]
            #     sf.sort(key=lambda t:t[0])
            #     print("macro:", sum([t[3] for t in sf]) / len(sf))
        # if not test_only:
        #     # score_fn = model.score
        #     score_fn = lambda batch: model.forward(batch, exemplar=False, exemplar_distill=False, distill=False,
        #                                 feature_distill=False, tau=0.5, task_id=loader_id, train_mode=False)
        #     # if worker.epoch == 1 or worker.epoch % 2 == 0:
        #     dev_loss, dev_metrics = worker.run_one_epoch(
        #             model=model,
        #             f_loss=score_fn,
        #             loader=loaders[-2],
        #             split="dev",
        #             collect_stats=collect_stats,
        #             collect_outputs=collect_outputs)
        #     dev_outputs = {k: torch.cat(v, dim=0) for k, v in worker.epoch_outputs.items()}
        #     dev_scores, (dev_p, dev_r, dev_f) = by_class(dev_outputs["prediction"], dev_outputs["label"],
        #                                                      learned_labels=learned_labels)
        #     dev_class_f1 = {k: dev_scores[k][2] for k in dev_scores}
        #     for k, v in dev_class_f1.items():
        #         add_summary_value(summary, f"dev_class_{k}", v, total_epoch)
        #     dev_metrics = dev_f
        #     for output_log in [print, worker._log]:
        #         output_log(
        #             f"Epoch {worker.epoch:3d}:  Dev {dev_metrics}"
        #         )
        #
        #     if best_dev is None or dev_metrics > best_dev:
        #         print("-----find best model on dev-----")
        #         best_dev = dev_metrics
        #         worker.save(model, optimizer, postfix=str(loader_id))   # save best model on dev
        #         # whether reset patient when a better dev found
        #         # no_better = 0
        #     else:
        #         no_better += 1
        #         print("-----hit patience-----")
        #     print(f"patience: {no_better} / {patience}")
        #
        #     if (no_better == patience) or (worker.epoch == worker.train_epoch) or (opts.skip_first and loader_id == 0) or (opts.skip_second and loader_id == 1):
        #         if no_better == patience:
        #             print("------early stop-----")
        #
        #         loader_id += 1
        #         no_better = 0
        #         worker.load(model, optimizer, path=os.path.join(opts.log_dir, f"{worker.save_model}.{loader_id - 1}"))
        #         score_fn = lambda batch: model.forward(batch, exemplar=False, exemplar_distill=False, distill=False,
        #                                                feature_distill=False, tau=0.5, task_id=loader_id-1,
        #                                                train_mode=False)
        #         if opts.task_type == 'ec':
        #             test_loader_tmp = test_loaders[loader_id]
        #         else:
        #             test_loader_tmp = test_loaders
        #         test_loss, test_metrics = worker.run_one_epoch(
        #             model=model,
        #             f_loss=score_fn,
        #             # loader=test_loaders,   # add [loader_id-1] or not
        #             # loader=test_loaders[loader_id-1],   # add [loader_id-1] or not
        #             loader=test_loader_tmp,
        #             split="test",
        #             collect_stats=collect_stats,
        #             collect_outputs=collect_outputs)
        #         test_outputs = {k: torch.cat(v, dim=0) for k, v in worker.epoch_outputs.items()}
        #         torch.save(test_outputs, f"./log/{os.path.basename(opts.load_model)}.output")
        #         test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"],
        #                                                          learned_labels=learned_labels)
        #         test_class_f1 = {k: test_scores[k][2] for k in test_scores}
        #         for k, v in test_class_f1.items():
        #             add_summary_value(summary, f"test_class_{k}", v, total_epoch)
        #
        #         test_metrics = test_f
        #         best_test = test_metrics
        #         print("-----Test F1-----")
        #         best_test = round(100 * best_test, 2)
        #         best_dev = round(100 * best_dev, 2)
        #         print(best_test)
        #
        #         best_dev_scores.append(best_dev)
        #         best_test_scores.append(best_test)
        #         print("-----------Current Best Dev Results----------")
        #         print(best_dev_scores)
        #         print("-----------Current Best Test Results----------")
        #         print(best_test_scores)
        #
        #         # TODO: switch of setting exemplar
        #         # if not opts.finetune:
        #         #     print("setting train exemplar for learned classes")
        #         #     model.set_exemplar(exemplar_loaders[loader_id-1], task_id=loader_id-1)
        #
        #         if opts.balance == "icarl":
        #             model.set_none_feat(loaders[loader_id-1])
        #         elif opts.balance == "bic" and loader_id >= 2:
        #             # train stream, release next stream
        #             # only apply to finish second round training (loader_id >= 1 + 1 = 2)
        #             print("setting dev exemplar for learned classes")
        #             model.set_exemplar(loaders[-2], q=5, label_sets=stage_labels[loader_id-1], output="dev")
        #             cur_dev_exe_f, cur_dev_exe_l= model.dev_exemplar_features, model.dev_exemplar_labels
        #             print("sample none instances for bic training")
        #             none_exemplar = model.set_exemplar(loaders[-2], q=int(none_mul*cur_dev_exe_f.size(0)), label_sets=[0], collect_none=True, output_only=True, output=None)
        #             cur_exe_f = torch.cat((none_exemplar[0], cur_dev_exe_f), dim=0)
        #             cur_exe_l = torch.cat((torch.zeros(none_exemplar[0].size(0)).to(cur_dev_exe_l), cur_dev_exe_l), dim=0)
        #             dev_bic_loader = DataLoader(
        #                 TensorDataset(cur_exe_f, cur_exe_l),
        #                 batch_size=128,
        #                 shuffle=True,
        #                 drop_last=False,
        #                 num_workers=1
        #                 )
        #             for _bias_epoch in range(2):
        #                 with torch.autograd.set_detect_anomaly(True):
        #                     worker.run_one_epoch(
        #                         model=model,
        #                         f_loss=model.forward_correction,
        #                         loader=dev_bic_loader,
        #                         split="train",
        #                         optimizer=optimizer_correction,
        #                         collect_stats=collect_stats,
        #                         prog=loader_id,
        #                         run='bic dev')
        #         elif opts.balance == "eeil" and loader_id >= 2:
        #             cur_exe_f, cur_exe_l= model.exemplar_features, model.exemplar_labels
        #             none_exemplar = model.set_exemplar(loaders[loader_id-1], q=int(none_mul*cur_exe_f.size(0)), label_sets=[0], collect_none=True, output_only=True, output=None)
        #             cur_exe_f = torch.cat((none_exemplar[0], cur_exe_f), dim=0)
        #             cur_exe_l = torch.cat((torch.zeros(none_exemplar[0].size(0)).to(cur_exe_l), cur_exe_l), dim=0)
        #             eeil_train_loader = DataLoader(
        #                 TensorDataset(cur_exe_f, cur_exe_l),
        #                 batch_size=128,
        #                 shuffle=True,
        #                 drop_last=False,
        #                 num_workers=1
        #                 )
        #             for i in range(5):
        #                 worker.run_one_epoch(
        #                     model=model,
        #                     f_loss=model.extra_forward,
        #                     loader=eeil_train_loader,
        #                     split="train",
        #                     optimizer=optimizer,
        #                     collect_stats=collect_stats,
        #                     prog=loader_id,
        #                     run='eeil balance')
        #         if opts.balance in ['eeil', 'bic']:
        #             worker.save(model, optimizer, postfix=str(loader_id-1))
        #             dev_loss, dev_metrics = worker.run_one_epoch(
        #                 model=model,
        #                 f_loss=model.score,
        #                 loader=loaders[-2],
        #                 split="dev",
        #                 collect_stats=collect_stats,
        #                 collect_outputs=collect_outputs)
        #             dev_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
        #             dev_scores, (dev_p, dev_r, dev_f) = by_class(dev_outputs["prediction"], dev_outputs["label"], learned_labels=learned_labels)
        #             dev_class_f1 = {k: dev_scores[k][2] for k in dev_scores}
        #             for k,v in dev_class_f1.items():
        #                 add_summary_value(summary, f"dev_class_{k}", v, total_epoch)
        #             dev_metrics = dev_f
        #             test_loss, test_metrics = worker.run_one_epoch(
        #                 model=model,
        #                 loader=loaders[-1],
        #                 f_loss=model.score,
        #                 split="test",
        #                 collect_stats=collect_stats,
        #                 collect_outputs=collect_outputs)
        #             test_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
        #             test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"], learned_labels=learned_labels)
        #             test_class_f1 = {k: test_scores[k][2] for k in test_scores}
        #             for k,v in test_class_f1.items():
        #                 add_summary_value(summary, f"test_class_{k}", v, total_epoch)
        #             test_metrics = test_f
        #             best_dev = dev_metrics; best_test = test_metrics
        #         if not opts.finetune:
        #             model.set_history()
        #         for output_log in [print, worker._log]:
        #             output_log(f"BEST DEV {loader_id-1}: {best_dev if best_dev is not None else 0}")
        #             output_log(f"BEST TEST {loader_id-1}: {best_test if best_test is not None else 0}")
        #         if loader_id == len(loaders) - 2:
        #             termination = True
        #         else:
        #             learned_labels = learned_labels.union(set(stage_labels[loader_id]))
        #             if opts.balance == 'bic':
        #                 model.correction_stream.append(max(learned_labels) + 1)
        #             if opts.kt:
        #                 next_exemplar = model.set_exemplar(exemplar_loaders[loader_id], output_only=True)
        #                 next_frequency = {}
        #                 indices = loaders[loader_id].dataset.label2index
        #                 for label in stage_labels[loader_id]:
        #                     if label != 0:
        #                         next_frequency[label] = indices[label]
        #                 if opts.kt2:
        #                     next_inits = model.initialize2(
        #                         exemplar=next_exemplar,
        #                         ninstances=next_frequency,
        #                         gamma=opts.kt_gamma,
        #                         tau=opts.kt_tau,
        #                         alpha=opts.kt_alpha,
        #                         delta=opts.kt_delta)
        #                 else:
        #                     next_inits = model.initialize(
        #                         exemplar=next_exemplar,
        #                         ninstances=next_frequency,
        #                         gamma=opts.kt_gamma,
        #                         tau=opts.kt_tau,
        #                         alpha=opts.kt_alpha)
        #                 torch.save(model.outputs["new2old"], os.path.join(opts.log_dir, f"{loader_id}_to_{loader_id-1}"))
        #                 model.extend(next_inits)
        #                 assert model.nslots == max(learned_labels) + 1
        #             else:
        #                 model.nslots = max(learned_labels) + 1
        #         worker.epoch = 0
        #         best_dev = None; best_test = None
    # print("-----------Dev Results----------")
    # print(best_dev_scores)
    # print("-----------Test Results----------")
    # print(best_test_scores)


if __name__ == "__main__":
    main()
