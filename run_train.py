import pdb
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from tqdm import tqdm
from utils.optimizer import AdamW
from utils.options import parse_arguments
# from utils.datastream import get_stage_loaders, get_stage_loaders_n
from contextlib import redirect_stdout

# from utils.dataloader_ace import get_stage_loaders, get_stage_loaders_n

from utils.worker import Worker
# from models.emp import PromptNet
# from models.adapter.prior_adapter import BERT
from utils.utils import get_task_stat

# from models.bert_baseline import BERT, BIC, ICARL
# from models.prev_baseline import BERT, BIC, ICARL

from models.sep_cls import SepCLS
# from models.sep_cls_ft import SepCLSFT

# from models.baseline import KDR,
import random

opts = parse_arguments()

if opts.task_type == 'ec' or opts.task_type == "rc":
    from utils.dataloader_no_other import get_stage_loaders, get_stage_loaders_n
else:
    from utils.dataloader import get_stage_loaders, get_stage_loaders_n

print(opts.save_model)
if not opts.ec_train_other:
    print("EC not train with other")
else:
    print("train with other")


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

    ag = 0; ad = 0; am = 0
    for label in range(1, nlabels+1):
        lg = (labels==label); ld = (preds==label)
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
    # summary = SummaryWriter(opts.log_dir)

    dataset_id = 0

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

    if opts.balance == "emp":
        model = PromptNet(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0]) + 1 if not opts.test_only else max(stage_labels[-1]) + 1,
            label_mapping=label2id,
            device=torch.device(
                torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
        )
    elif opts.balance == "sepcls_all_prev" or opts.balance == "sepcls_individual":
        model = SepCLS(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0]) + 1 if not opts.test_only else max(stage_labels[-1]) + 1,
            label_mapping=label2id,
            device=torch.device(
                torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
        )
    elif opts.balance == "sepcls_ft":
        model = SepCLSFT(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0]) + 1 if not opts.test_only else max(stage_labels[-1]) + 1,
            label_mapping=label2id,
            device=torch.device(
                torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
        )
    else:
        model = BERT(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0])+1 if not opts.test_only else max(stage_labels[-1])+1,
            label_mapping=label2id,
            device=torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
        )
    param_groups = [
        {"params": [param for name, param in model.named_parameters() if param.requires_grad and 'correction' not in name],
        "lr":lr,
        "weight_decay": opts.decay,
        "betas": (0.9, 0.999)}
        ]
    optimizer = AdamW(params=param_groups)
    worker = Worker(opts)
    worker._log(str(opts))
    worker._log(str(label2id))
    if opts.test_only:
        worker.load(model, path=opts.model_dir)

    best_dev = best_test = None
    collect_stats = "accuracy"
    collect_outputs = {"prediction", "label"}
    termination = False
    patience = opts.patience
    no_better = 0
    loader_id = 0
    # if opts.resume:
    #     loader_id = opts.resume_loader_id

    total_epoch = 0
    none_mul = 4
    learned_labels = set(stage_labels[0])
    best_dev_scores = []
    best_test_scores = []
    per_type_f1_list = []
    dev_metrics = None
    test_metrics = None
    # exemplar_flag = opts.replay_flag
    if opts.balance in ['eeil', 'bic', 'kcn', 'kt', 'emp', 'replay'] or opts.replay_flag_bool:
        exemplar_flag = True
    else:
        exemplar_flag = False
    print(f"replay flag: {exemplar_flag}")

    while not termination:
        if not opts.test_only:
            if opts.skip_first and loader_id == 0:
                worker.load(model, optimizer, path=opts.load_first, strict=opts.balance!='bic')
                total_epoch += worker.epoch
            elif opts.skip_second and loader_id == 1:
                worker.load(model, optimizer, path=opts.load_second, strict=opts.balance!='bic')
                total_epoch += worker.epoch
            else:
                if opts.finetune:
                    train_loss = lambda batch:model.forward(batch)
                elif opts.balance == "kcn":
                    train_loss = lambda batch:model.forward(batch, exemplar=exemplar_flag, feature_distill=True, exemplar_distill=exemplar_flag, distill=True, tau=0.5, task_id=loader_id)
                elif opts.balance == "kt":
                    train_loss = lambda batch:model.forward(batch, exemplar=exemplar_flag, mul_distill=True, exemplar_distill=exemplar_flag, distill=True, tau=0.5, task_id=loader_id)
                elif opts.balance == "kd":
                    train_loss = lambda batch:model.forward(batch, exemplar=False, mul_distill=True, exemplar_distill=False, distill=True, tau=0.5, task_id=loader_id)
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

        if opts.test_only:
            score_fn = model.score

            test_loss, test_metrics = worker.run_one_epoch(
                model=model,
                f_loss=score_fn,
                loader=loaders[-1],
                split="test",
                collect_stats=collect_stats,
                collect_outputs=collect_outputs)
            test_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
            torch.save(test_outputs, f"log/{os.path.basename(opts.load_model)}.output")
            test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"], learned_labels=learned_labels)
            test_class_f1 = {k: test_scores[k][2] for k in test_scores}
            # for k,v in test_class_f1.items():
            #     add_summary_value(summary, f"test_class_{k}", v, total_epoch)
            test_metrics = test_f
            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}: Test {test_metrics}"
                )

        if not opts.test_only:
            # score_fn = model.score
            score_fn = lambda batch: model.forward(batch, exemplar=False, exemplar_distill=False, distill=False,
                                        feature_distill=False, tau=0.5, task_id=loader_id, train_mode=False)
            # if worker.epoch == 1 or worker.epoch % 2 == 0:
            if opts.task_type == 'ec' or opts.task_type == "rc":
                dev_loader_tmp = dev_loaders[loader_id]
            else:
                dev_loader_tmp = dev_loaders
            dev_loss, dev_metrics = worker.run_one_epoch(
                    model=model,
                    f_loss=score_fn,
                    loader=dev_loader_tmp,  # add [loader_id] or not
                    split="dev",
                    collect_stats=collect_stats,
                    collect_outputs=collect_outputs)
            print("Non select num in dev: ", model.non_select_cnt)
            dev_outputs = {k: torch.cat(v, dim=0) for k, v in worker.epoch_outputs.items()}
            dev_scores, (dev_p, dev_r, dev_f) = by_class(dev_outputs["prediction"], dev_outputs["label"],
                                                             learned_labels=learned_labels)
            dev_class_f1 = {k: dev_scores[k][2] for k in dev_scores}
            # for k, v in dev_class_f1.items():
            #     add_summary_value(summary, f"dev_class_{k}", v, total_epoch)
            dev_metrics = dev_f
            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}:  Dev {dev_metrics}"
                )

            if best_dev is None or dev_metrics > best_dev:
                print("-----find best model on dev-----")
                best_dev = dev_metrics
                worker.save(model, optimizer, postfix=str(loader_id))   # save best model on dev
                # whether reset patient when a better dev found
                # no_better = 0
            else:
                no_better += 1
                print("-----hit patience-----")
            print(f"patience: {no_better} / {patience}")

            if (no_better == patience) or (worker.epoch == worker.train_epoch) or (opts.skip_first and loader_id == 0) or (opts.skip_second and loader_id == 1):
                if no_better == patience:
                    print("------early stop-----")

                loader_id += 1
                no_better = 0
                worker.load(model, optimizer, path=os.path.join(opts.log_dir, f"{worker.save_model}.{loader_id - 1}"))
                score_fn = lambda batch: model.forward(batch, exemplar=False, exemplar_distill=False, distill=False,
                                                       feature_distill=False, tau=0.5, task_id=loader_id-1,
                                                       train_mode=False)
                if opts.task_type == 'ec' or opts.task_type == "rc":
                    test_loader_tmp = test_loaders[loader_id-1]
                else:
                    test_loader_tmp = test_loaders
                test_loss, test_metrics = worker.run_one_epoch(
                    model=model,
                    f_loss=score_fn,
                    loader=test_loader_tmp,
                    split="test",
                    collect_stats=collect_stats,
                    collect_outputs=collect_outputs)
                print("Non select num in test: ", model.non_select_cnt)
                test_outputs = {k: torch.cat(v, dim=0) for k, v in worker.epoch_outputs.items()}
                torch.save(test_outputs, f"./log/{os.path.basename(opts.load_model)}.output")
                test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"],
                                                                 learned_labels=learned_labels)
                test_class_f1 = {k: test_scores[k][2] for k in test_scores}
                print("------F1 per class------")

                print(json.dumps(test_class_f1))

                # for k, v in test_class_f1.items():
                #     add_summary_value(summary, f"test_class_{k}", v, total_epoch)

                per_type_f1_list.append(test_class_f1)

                test_metrics = test_f
                best_test = test_metrics
                print("-----Test F1-----")
                best_test = round(100 * best_test, 2)
                best_dev = round(100 * best_dev, 2)
                print(best_test)

                best_dev_scores.append(best_dev)
                best_test_scores.append(best_test)
                print("-----------Current Best Dev Results----------")
                print(best_dev_scores)
                print("-----------Current Best Test Results----------")
                print(best_test_scores)

                # TODO: switch of setting exemplar
                if exemplar_flag:
                    print("setting train exemplar for learned classes")
                    model.set_exemplar(exemplar_loaders[loader_id-1], task_id=loader_id-1)

                # set prompt's require_grad
                if opts.balance == "emp":
                    model.prompted_embed.prompt_list[loader_id+1].requires_grad = True

                if opts.balance == "sepcls_all_prev" or opts.balance == "sepcls_individual" or opts.balance == "sepcls_ft":
                    for name, param in list(model.sep_classifier[loader_id].named_parameters()):
                        param.requires_grad = False

                    worker.save(model, optimizer, postfix=str(loader_id-1))
                    dev_loss, dev_metrics = worker.run_one_epoch(
                        model=model,
                        f_loss=model.score,
                        loader=loaders[-2],
                        split="dev",
                        collect_stats=collect_stats,
                        collect_outputs=collect_outputs)
                    dev_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
                    dev_scores, (dev_p, dev_r, dev_f) = by_class(dev_outputs["prediction"], dev_outputs["label"], learned_labels=learned_labels)
                    dev_class_f1 = {k: dev_scores[k][2] for k in dev_scores}
                    # for k,v in dev_class_f1.items():
                    #     add_summary_value(summary, f"dev_class_{k}", v, total_epoch)
                    dev_metrics = dev_f
                    test_loss, test_metrics = worker.run_one_epoch(
                        model=model,
                        loader=loaders[-1],
                        f_loss=model.score,
                        split="test",
                        collect_stats=collect_stats,
                        collect_outputs=collect_outputs)
                    test_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
                    test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"], learned_labels=learned_labels)
                    test_class_f1 = {k: test_scores[k][2] for k in test_scores}

                    # for k,v in test_class_f1.items():
                    #     add_summary_value(summary, f"test_class_{k}", v, total_epoch)
                    test_metrics = test_f
                    best_dev = dev_metrics; best_test = test_metrics
                                  
                if not opts.finetune:
                    model.set_history()
                for output_log in [print, worker._log]:
                    output_log(f"BEST DEV {loader_id-1}: {best_dev if best_dev is not None else 0}")
                    output_log(f"BEST TEST {loader_id-1}: {best_test if best_test is not None else 0}")

                # if loader_id == len(loaders) - 2:
                #     termination = True
                if loader_id == len(loaders):
                    termination = True
                else:
                    learned_labels = learned_labels.union(set(stage_labels[loader_id]))
                    if opts.kt:
                        next_exemplar = model.set_exemplar(exemplar_loaders[loader_id], output_only=True)
                        next_frequency = {}
                        indices = loaders[loader_id].dataset.label2index
                        for label in stage_labels[loader_id]:
                            if label != 0:
                                next_frequency[label] = indices[label]
                        if opts.kt2:
                            next_inits = model.initialize2(
                                exemplar=next_exemplar,
                                ninstances=next_frequency,
                                gamma=opts.kt_gamma,
                                tau=opts.kt_tau,
                                alpha=opts.kt_alpha,
                                delta=opts.kt_delta)
                        else:
                            next_inits = model.initialize(
                                exemplar=next_exemplar,
                                ninstances=next_frequency,
                                gamma=opts.kt_gamma,
                                tau=opts.kt_tau,
                                alpha=opts.kt_alpha)
                        torch.save(model.outputs["new2old"], os.path.join(opts.log_dir, f"{loader_id}_to_{loader_id-1}"))
                        model.extend(next_inits)
                        assert model.nslots == max(learned_labels) + 1
                    else:
                        model.nslots = max(learned_labels) + 1
                worker.epoch = 0
                best_dev = None; best_test = None
    new_f1s, acc_old_f1s, accumulate_old_type_f1_per_task = new_and_old(per_type_f1_list)

    print(f"Task Permutation: {opts.perm_id}")
    print("-----------Dev Results----------")
    print(best_dev_scores)
    print("-----------Test Results----------")
    print(best_test_scores)

    with open(f'outputs/{opts.eval_model_name}.txt', 'a+') as f:
        with redirect_stdout(f):
            print(f"Task Permutation: {opts.perm_id}")
            print("-----------Dev Results----------")
            print(best_dev_scores)
            print("-----------Test Results----------")
            print(best_test_scores)
            print("New Type F1:")
            print(new_f1s)
            print("Accumulate Old Type F1:")
            print(acc_old_f1s)
            print("Per Task Old Type F1:")
            print(accumulate_old_type_f1_per_task)

if __name__ == "__main__":
    main()
