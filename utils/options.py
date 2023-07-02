import argparse
import os
import glob

model_names = ['bert_large', 'debug', 'naive_prompt', 'task_prompt', 'generic_prompt']
log_names = ['prompt_log', 'bert_log', 'debug_log', 'replay_log', 'll_log']

dataset_names = ["MAVEN", "ACE", "FewNERD", "TACRED", "FewRel"]
task_types = ["ed", "ec", "re", "rc"]
# global reconstruction loss variable to collect loss in adapter
# RECONSTRUCT_LOSS = 0

dataset = "MAVEN"
variant = "sepcls_individual"
task = "ed"
eval_model_name = "maven_ed_oth_logit_tune"
gpu_id = 7
perm_id = 0
ec_train_other = 0
oth_logit = 0


# ------------------------------------
eval_loader_id = 0
if task =="ec" or task =="rc":
    lr = 5e-4
elif variant == "sepcls_ft":
    lr = 1e-6
else:
    lr = 1e-4
if dataset == "MAVEN":
    num_classes = 169
    init_slot = 34
elif dataset == "ACE":
    num_classes = 34
    init_slot = 10
elif dataset == "FewNERD":
    num_classes = 67
    init_slot = 13
    # init_slot = 7        # for super type setting
elif dataset == "TACRED":
    num_classes = 41
    init_slot = 5

if perm_id == 5:
    init_slot = num_classes
# print(f"{dataset}_{variant}_0")

def define_arguments(parser):
    parser.add_argument('--json-root', type=str, default="./data", help="")
    parser.add_argument('--dataset', type=str, default=dataset, help="")
    parser.add_argument('--task-type', type=str, default=task, help="")
    parser.add_argument('--feature-root', type=str, default=f"./data/features/{dataset}/", help="")
    # parser.add_argument('--stream-file', type=str, default=f"data/{dataset}/onestream.json", help="")
    parser.add_argument('--stream-file', type=str, default=f"data/{dataset}/streams.json", help="")
    parser.add_argument('--ae-dir-path', type=str, default="./ae_ckpt/", help="")
    # parser.add_argument('--stream-file', type=str, default="data/ACE/streams.json", help="")
    parser.add_argument('--batch-size', type=int, default=1, help="")
    # parser.add_argument('--init-slots', type=int, choices=[34, 10], default=init_slot, help="")
    parser.add_argument('--grad-accumulate-step', type=int, default=8, help="")
    parser.add_argument('--max-slots', type=int, choices=[169, 34, 67, 41], default=num_classes, help="")
    parser.add_argument('--prompt-size', type=int, choices=[169, 34, 67, 41], default=num_classes, help="")
    parser.add_argument('--nhead', type=int, default=8, help="")
    parser.add_argument('--nlayers', type=int, default=3, help="")
    parser.add_argument('--no-gpu', action="store_true", help="don't use gpu")
    parser.add_argument('--decay', type=float, default=1e-2, help="")
    parser.add_argument('--tau', type=float, default=0.5, help="")
    parser.add_argument('--kt-alpha', type=float, default=0.25, help="")
    parser.add_argument('--kt-gamma', type=float, default=0.05, help="")
    parser.add_argument('--kt-tau', type=float, default=1.0, help="")
    parser.add_argument('--kt-delta', type=float, default=0.5, help="")
    parser.add_argument('--ae_score_temp', type=float, default=1, help="")
    parser.add_argument('--seed', type=int, default=2147483647, help="random seed")
    #---------------------------------------------------
    parser.add_argument('--input-dim', type=int, default=2048, help="")
    parser.add_argument('--hidden-dim', type=int, default=1024, help="")
    parser.add_argument('--patience', type=int, choices=[4, 5, 6], default=3, help="")
    parser.add_argument('--gpu', type=int, default=gpu_id, help="gpu")
    parser.add_argument('--device', type=str, default=f"cuda:{gpu_id}", help="device")
    parser.add_argument('--learning-rate', type=float, choices=[1e-3, 1e-4, 5e-4, 5e-6, 1e-6, 5e-5, 1e-5], default=lr, help="")
    parser.add_argument('--perm-id', type=int, default=perm_id, help="")
    parser.add_argument('--save-model', type=str, default=f"{eval_model_name}", help="checkpoints name")
    parser.add_argument('--load-model', type=str, default=f"{eval_model_name}", help="path to saved checkpoint")
    parser.add_argument('--log-dir', type=str, default=f"logs/{eval_model_name}", help="path to save log file")
    parser.add_argument('--replay-flag', action="store_true", help='')
    parser.add_argument('--replay-flag-bool', type=int, default=0, help='')

    parser.add_argument('--matching-loss-w', type=int, default=0.1, help="")
    parser.add_argument('--negative-fraction', type=int, default=10, help="")
    parser.add_argument('--kt', action="store_true", help='')
    parser.add_argument('--full_negative', action="store_true", help='')
    parser.add_argument('--balance', choices=['icarl', 'eeil', 'bic', 'none', 'kcn', 'kt', 'nod', 'emp', 'sepcls_all_prev', 'sepcls_individual', 'replay', 'kd', 'ffe', 'sepcls_ft', "sepcls"], default=variant)
    #---------------------------------------------------
    # parser.add_argument('--model-dir', type=str, default=f"logs/{eval_model_name}/{eval_model_name}.{eval_loader_id}", help="path to load model checkpoint")
    parser.add_argument('--model-dir', type=str, default="", help="path to load model checkpoint")
    parser.add_argument('--eval-model-name', type=str, default=f"{eval_model_name}", help="path to load model checkpoint")
    parser.add_argument('--eval-loader-id', type=int, default=eval_loader_id, help='loader id for eval')

    parser.add_argument('--train-epoch', type=int, default=15, help='epochs to train')
    parser.add_argument('--period', type=int, default=8, help='exemplar replay interval')
    parser.add_argument('--eloss_w', type=int, default=50, help='exemplar replay loss weight')
    parser.add_argument('--kt2', action="store_true", help='')
    parser.add_argument('--finetune', action="store_true", help='')
    parser.add_argument('--load-first', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--skip-first', action="store_true", help='')
    parser.add_argument('--load-second', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--skip-second', action="store_true", help='')
    parser.add_argument('--test-only', action="store_true", help='is testing')
    parser.add_argument('--case-study', type=int, default=0, help='')
    parser.add_argument('--ec-train-other', type=int, default=ec_train_other, help='')
    parser.add_argument('--setting', choices=['classic', "new"], default="classic")

    parser.add_argument('--def-other-logit', type=int, default=oth_logit, help='')


def parse_arguments():
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    args = parser.parse_args()
    args.log = os.path.join(args.log_dir, "logfile.log")
    if (not args.test_only) and os.path.exists(args.log_dir):
        existing_logs = glob.glob(os.path.join(args.log_dir, "*"))
        for _t in existing_logs:
            os.remove(_t)
    return args
