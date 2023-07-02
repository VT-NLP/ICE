import json
import numpy
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch import autograd
from transformers import BertModel, BertConfig, BertTokenizerFast
import math
from typing import Any, Dict, Tuple, List, Union, Set
import warnings
from collections import OrderedDict
from torch.nn.modules.linear import Linear
from torchmeta.modules import MetaLinear, MetaSequential, MetaModule, MetaBilinear
from tqdm import tqdm
from utils.options import parse_arguments
import random
import pdb
import torch.nn.functional as F
from utils.utils import get_task_stat

opts = parse_arguments()
BERT_VOCAB_SIZE = 28996
BERT_MAXLEN = 512

PERM, TASK_NUM, TASK_EVENT_NUM, NA_TASK_EVENT_NUM, ACC_NUM = get_task_stat(opts.dataset, opts.perm_id)

device = torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))

file_path = f"./data/{opts.dataset}/id2tokens.json"
label2id_path = f"./data/{opts.dataset}/label2id.json"

streams = json.load(open(opts.stream_file))
id2label_dict = {}
with open(label2id_path, 'rt') as fp:
    label2id = json.load(fp)
for label, idx in label2id.items():
    task_id = 0
    for i in range(len(streams)):
        if idx in streams[i]:
            task_id = i
            break
    id2label_dict[label2id[label]] = label + "(" + str(task_id) + ")"
id2label_dict[0] = "Other"

random.seed(opts.seed)


class SepCLSFT(MetaModule):
    def __init__(self, input_dim: int, hidden_dim: int, max_slots: int, init_slots: int, label_mapping: dict,
                 device: Union[torch.device, None] = None, task_id=0, **kwargs) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(opts.bert_model)
        self.half_input_dim = int(input_dim / 2)

        if opts.task_type == "re" or opts.task_type == "rc":
            k = 2
        else:
            k = 1

        self.sep_classifier = nn.ModuleList([nn.Linear(hidden_dim*k*2, NA_TASK_EVENT_NUM[i], bias=False) for i in range(TASK_NUM+1)])
        _mask = torch.zeros(1, max_slots, dtype=torch.float, device=device)
        _mask[:, init_slots:] = float("-inf")
        self.register_buffer(name="_mask", tensor=_mask)
        self.crit = nn.CrossEntropyLoss()
        self.device = device
        self.to(device=device)
        self.nslots = init_slots
        self.max_slots = max_slots
        self.maml = True
        self.outputs = {"input_ids": []}
        self.history = None
        self.exemplar_input = None
        self.exemplar_attm = None
        self.exemplar_labels = None
        self.exemplar_span = None
        self.exemplar_logit = None
        self.exemplar_feature = None
        self.exemplar_size = None
        self.random_exemplar_inx = None

        self.analysis_out = {"sentence_ids": [], "sentences": [], "triggers": [], "labels": [], "predictions": [],
                             "pred_logits": [], "gold_logits": []}
        self.id2label = {}
        for label in label_mapping.keys():
            self.id2label[label_mapping[label]] = label
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")

        self.iter_cnt = 0
        self.period = opts.period
        self.e_weight = opts.eloss_w

        max_id = ACC_NUM[task_id+1]
        if opts.task_type == "ec" or opts.task_type == "rc":
            max_id -= 1
        self.mean_task_logit = torch.zeros(max_id, dtype=torch.float, device=device)
        self.mean_task1_logit = torch.zeros(max_id, dtype=torch.float, device=device)
        self.mean_task_prob = torch.zeros(max_id, dtype=torch.float, device=device)
        self.mean_task1_prob = torch.zeros(max_id, dtype=torch.float, device=device)

        self.task_instance_cnt = 0
        self.task1_instance_cnt = 0
        self.span_cnt = 0



    @property
    def mask(self, ):
        self._mask[:, :self.nslots] = 0
        self._mask[:, self.nslots:] = float("-inf")
        return self._mask

    def idx_mask(self, idx: Union[torch.LongTensor, int, List[int], None] = None,
                 max_idx: Union[torch.LongTensor, int, None] = None):
        assert (idx is not None) or (max_idx is not None)
        assert (idx is None) or (max_idx is None)
        mask = torch.zeros_like(self._mask) + float("-inf")
        if idx is not None:
            mask[:, idx] = 0
        if max_idx is not None:
            if isinstance(max_idx, torch.LongTensor):
                max_idx = max_idx.item()
            mask[:, :max_idx] = 0
        return mask


    def mask_train(self, task_id):
        self._mask = torch.zeros_like(self._mask) + float("-inf")
        self._mask[:, ACC_NUM[task_id]:ACC_NUM[task_id+1]] = 0
        return self._mask

    def idx_mask_train(self, task_id, idx: Union[torch.LongTensor, int, List[int], None] = None,
                 max_idx: Union[torch.LongTensor, int, None] = None):
        mask = torch.zeros_like(self._mask) + float("-inf")
        mask[:, ACC_NUM[task_id]:ACC_NUM[task_id+1]] = 0
        return mask

    @property
    def features(self):
        return self.classes.weight[:self.nslots]

    @staticmethod
    def avg_span(encoding, span_mask):
        s_mask = span_mask.unsqueeze(1)
        span_len = (span_mask != 0).sum(dim=1).unsqueeze(1)
        s_sum = torch.bmm(s_mask.float(), encoding).squeeze(1)
        s_avg = s_sum.float() / span_len.float()
        return s_avg

    @staticmethod
    def select_span_feature(features, spans):
        span_bsz = spans.size(0)

        feat = features.repeat(span_bsz, 1, 1)
        span_idx = spans.unsqueeze(-1)
        span_idx = span_idx.expand(-1, -1, 1024)

    def convert_label_id_to_token(self, label_list):
        types = [id2label_dict[self.id2label[i]] for i in label_list]
        return types

    def forward(self, batch, nslots: int = -1, exemplar: bool = False, exemplar_distill: bool = False,
                feature_distill: bool = False, mul_distill=False, distill: bool = False, return_loss: bool = True,
                return_feature: bool = False, tau: float = 1.0, log_outputs: bool = True, params=None, task_id: int = 0,
                store: bool = False, train_mode: bool=False):

        self.iter_cnt += 1
        input_ids, attention_masks, labels, spans, bert_feat, _ = batch.token_ids, batch.attention_masks, batch.labels, \
                                                                              batch.spans, batch.features, batch.prompt_masks

        if opts.case_study:
            # check if contain the instance in current stream (eval only)
            max_id = self.nslots
            if torch.any((labels < max_id) & (labels > 0)):
                curr_stream = True
                task_mask = torch.where((labels < max_id) & (labels > 0))
                task_one_mask = torch.where((labels < 34) & (labels > 0))
                self.span_cnt += task_mask[0].shape[0]
            else:
                curr_stream = False


        span_bsz = spans.size(0)
        if store:
            span_input = input_ids.unsqueeze(0).repeat(span_bsz, 1, 1)
            span_attm = attention_masks.repeat(span_bsz, 1, 1)
            self.outputs["input"] = span_input.detach().cpu()
            self.outputs["attm"] = span_attm.detach().cpu()
        self.outputs["input_ids"].append(input_ids)

        if opts.case_study and curr_stream:
            types = self.convert_label_id_to_token(labels.tolist())
            self.analysis_out["labels"].append(types)
            self.analysis_out["sentence_ids"].append(batch.meta["sentence_ids"])
            span_list = spans.tolist()
            token_list = input_ids.tolist()
            trigger_list = []
            for s in span_list:
                if s[0] == s[1]:
                    trigger_id = token_list[s[0]]
                else:
                    trigger_id = token_list[s[0]:s[1]]
                trigger = self.tokenizer.decode(trigger_id)
                trigger_list.append(trigger)
            self.analysis_out["triggers"].append(trigger_list)
            raw_sentence = self.tokenizer.decode(token_list).split('[SEP]')
            self.analysis_out["sentences"].append(raw_sentence[0][6:])

        # attention_masks = attention_masks[task_id]

        outputs = self.bert(input_ids.unsqueeze(0), attention_mask=attention_masks.unsqueeze(0))
        enc_outputs = outputs[0]

        # enc_outputs = bert_feat

        bsz, seq_len, hidden_dim = enc_outputs.shape
        span_bsz = spans.size(0)
        rep_enc_outputs = enc_outputs.repeat(span_bsz, 1, 1)

        if opts.task_type == "re" or opts.task_type == "rc":
            span1, span2 = torch.split(spans, 1, dim=1)
            span1 = span1.squeeze(1)
            span2 = span2.squeeze(1)
            span_idx1 = span1.unsqueeze(-1)
            span_idx1 = span_idx1.expand(-1, -1, hidden_dim)
            span_repr1 = torch.gather(rep_enc_outputs, 1, span_idx1)
            # pdb.set_trace()
            features1 = span_repr1.view(span_bsz, hidden_dim * 2)

            span_idx2 = span2.unsqueeze(-1)
            span_idx2 = span_idx2.expand(-1, -1, hidden_dim)
            span_repr2 = torch.gather(rep_enc_outputs, 1, span_idx2)
            features2 = span_repr2.view(span_bsz, hidden_dim * 2)

            features = torch.cat([features1, features2], dim=-1)
        else:
            span_idx = spans.unsqueeze(-1)
            span_idx = span_idx.expand(-1, -1, hidden_dim)
            span_repr = torch.gather(rep_enc_outputs, 1, span_idx)
            # input feature: (span_bsz, hidden_dim*2)
            features = span_repr.view(span_bsz, hidden_dim * 2)

        # original scores
        inputs = features

        cls_in = features
        # ----------- separated classifier ---------
        scores = self.sep_classifier[0](cls_in)
        for i in range(1, TASK_NUM+1):
            temp_score = self.sep_classifier[i](cls_in)
            scores = torch.cat((scores, temp_score), dim = -1)
        # -----------

        # scores = self.classes(inputs, params=self.get_subdict(params, "classes"))

        # ------------------------------------------------
        if opts.balance == "sepcls_all_prev":
            if torch.any(torch.isnan(scores)):
                print(scores[0])
                input('a')
            if nslots == -1:
                scores += self.mask
                if torch.any(torch.isnan(scores)):
                    print(scores[0])
                    input()
                nslots = self.nslots
            else:
                scores += self.idx_mask(max_idx=nslots)
        # ------------------------------------------------
        elif opts.balance == "sepcls_individual" or opts.balance == "sepcls_ft":
            if train_mode:
                if nslots == -1:
                    scores += self.mask_train(task_id)
                    if torch.any(torch.isnan(scores)):
                        print(scores[0])
                        input()
                    nslots = self.nslots
                else:
                    scores += self.idx_mask_train(task_id)
            else:
                if nslots == -1:
                    scores += self.mask
                    if torch.any(torch.isnan(scores)):
                        print(scores[0])
                        input()
                    nslots = self.nslots
                else:
                    scores += self.idx_mask(max_idx=nslots)

        if opts.task_type != "ec" and opts.task_type != "rc":
            scores[:, 0] = 0
            start_idx = 0
        else:
            scores[:, 0] = float("-inf")
            start_idx = 1
            if train_mode and opts.ec_train_other:
                scores[:, 0] = 0

        # scores[:, 0] = 0
        # scores[:, 0] = float("-inf")    # TODO (neg)
        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        else:
            labels = labels
        if log_outputs:
            pred = torch.argmax(scores, dim=1)
            acc = torch.mean((pred == labels).float())
            if opts.case_study and curr_stream:
                select_score = scores[task_mask]
                avg_logit = torch.mean(select_score[:, start_idx:max_id], dim=0)
                avg_prob = torch.mean(F.softmax(select_score[:, start_idx:max_id], dim=-1), dim=0)
                self.mean_task_logit += avg_logit
                self.mean_task_prob += avg_prob
                if torch.any((labels < 34) & (labels > 0)):
                    select_score_t1 = scores[task_one_mask]
                    avg_logit_t1 = torch.mean(select_score_t1[:, start_idx:max_id], dim=0)
                    avg_prob_t1 = torch.mean(F.softmax(select_score_t1[:, start_idx:max_id], dim=-1), dim=0)
                    self.mean_task1_logit += avg_logit_t1
                    self.mean_task1_prob += avg_prob_t1
                    self.task1_instance_cnt += 1
                self.task_instance_cnt += 1
                pred_logits = torch.max(scores, dim=1).values
                pred_logits = [round(i, 2) for i in pred_logits.tolist()]
                gold_logits = torch.diagonal(scores[:, labels])
                gold_logits = [round(i, 2) for i in gold_logits.tolist()]
                other_logits = scores[:, 0]
                other_logits = [round(i, 2) for i in other_logits.tolist()]
                preds_labels = self.convert_label_id_to_token(pred.tolist())
                self.analysis_out["predictions"].append(preds_labels)
                self.analysis_out["pred_logits"].append(pred_logits)
                self.analysis_out["gold_logits"].append(gold_logits)
                # self.analysis_out["other_logits"].append(other_logits)
            self.outputs["accuracy"] = acc.item()
            self.outputs["prediction"] = pred.detach().cpu()
            self.outputs["logit"] = scores.detach().cpu()
            # self.outputs["attm"] = span_attm.detach().cpu()
            self.outputs["label"] = labels.detach().cpu()
            self.outputs["spans"] = spans.detach().cpu()
            self.outputs["encoded_features"] = inputs.detach().cpu()

        # if not train_mode:
        #     print(labels)
        #     pdb.set_trace()

        if return_loss:
            labels.masked_fill_(labels >= nslots, 0)
            valid = labels < nslots
            # TODO (neg): labels < nslots
            # valid_other = labels > 0
            # valid = torch.logical_and(valid, valid_other)

            nvalid = torch.sum(valid.float())
            # pdb.set_trace()
            if nvalid == 0:
                loss = 0
            else:
                loss = self.crit(scores[valid], labels[valid])  #
                if torch.isnan(loss):
                    print(labels, nslots, scores[:, :nslots])
                    input()
            if distill and self.history is not None:
                old_scores, old_inputs = self.forward(batch, nslots=self.history["nslots"], return_loss=False,
                                                      log_outputs=False, return_feature=True,
                                                      params=self.history["params"])
                self.iter_cnt -= 1
                old_scores = old_scores.detach()
                old_inputs = old_inputs.detach()
                new_scores = scores[:, :self.history["nslots"]]
                if mul_distill:
                    loss_distill = - torch.sum(
                        torch.softmax(old_scores * tau, dim=1) * torch.log_softmax(new_scores * tau, dim=1),
                        dim=1).mean()
                    old_dist = torch.softmax(old_scores / tau, dim=1)
                    old_valid = (old_dist[:, 0] < 0.9)
                    old_num = torch.sum(old_valid.float())
                    if old_num > 0:
                        # print(old_dist[old_valid].topk(5, dim=1), batch.labels[old_valid])
                        # input()
                        loss_mul_distill = - torch.sum(
                            old_dist[old_valid] * torch.log_softmax(new_scores[old_valid], dim=1), dim=1).sum()
                        loss_distill = (loss_distill * old_dist.size(0) + loss_mul_distill) / (
                                    old_dist.size(0) + old_num)
                        # loss_distill = loss_mul_distill
                else:
                    loss_distill = - torch.sum(
                        torch.softmax(old_scores * tau, dim=1) * torch.log_softmax(new_scores * tau, dim=1),
                        dim=1).mean()
                if feature_distill:
                    loss_f_distill = (1 - (
                                old_inputs / old_inputs.norm(dim=-1, keepdim=True) * inputs / inputs.norm(dim=-1,
                                                                                                          keepdim=True)).sum(
                        dim=-1)).mean(dim=0)
                    loss_distill += loss_f_distill

                d_weight = self.history["nslots"]
                c_weight = (self.nslots - self.history["nslots"])
                loss = (d_weight * loss_distill + c_weight * loss) / (d_weight + c_weight)
                if torch.isnan(loss):
                    print(old_scores, new_scores)
                    input()

            if exemplar and self.exemplar_input is not None and self.iter_cnt % self.period == 0:
                idx = self.random_exemplar_inx[(int(self.iter_cnt/self.period) - 1) % self.exemplar_size]
                example_feature, exemplar_scores = self.replay_forward(params, task_id, idx)
                exemplar_scores[:, 0] = 0.
                label = self.exemplar_labels[idx].to(self.device).unsqueeze(0)
                loss_exemplar = self.crit(exemplar_scores + self.mask, label)
                if torch.isnan(loss_exemplar):
                    print(self.exemplar_labels, nslots)
                    input()
                if exemplar_distill:
                    exemplar_old_scores = self.exemplar_logit[idx].unsqueeze(0).to(self.device)
                    example_old_feature = self.exemplar_feature[idx].unsqueeze(0).to(self.device)
                    # example_old_feature, exemplar_old_scores = self.replay_forward(self.history["params"], task_id, idx)
                    exemplar_old_scores[:, 0] = 0.
                    exemplar_old_scores = exemplar_old_scores[:self.history["nslots"]]
                    loss_exemplar_distill = - torch.sum(
                        torch.softmax(exemplar_old_scores[:self.history["nslots"]] * tau, dim=1) * torch.log_softmax(
                            exemplar_scores[:self.history["nslots"]], dim=1), dim=1).mean()
                    # distill CLS token
                    if feature_distill:
                        loss_exemplar_feat_distill = (1 - (example_old_feature / example_old_feature.norm(dim=-1,
                                                                                                          keepdim=True) * example_feature / example_feature.norm(
                            dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
                        loss_exemplar_distill += loss_exemplar_feat_distill
                    d_weight = self.history["nslots"]
                    c_weight = (self.nslots - self.history["nslots"])
                    loss_exemplar = (d_weight * loss_exemplar_distill + c_weight * loss_exemplar) / (
                                d_weight + c_weight)

                e_weight = self.exemplar_labels.size(0)
                # e_weight = 1
                loss = (nvalid * loss + e_weight * loss_exemplar) / (nvalid + e_weight)
                if torch.isnan(loss):
                    print(loss, loss_exemplar)
            return loss
        else:
            if return_feature:
                return scores[:, :nslots], inputs
            else:
                return scores[:, :nslots]

    def replay_forward(self, params, task_id, idx):
        # num_exem_iter = self.exemplar_labels.size(0)
        # exemplar_scores = []
        # for i in range(num_exem_iter):
        # i = self.random_exemplar_inx[(self.iter_cnt-1) % self.exemplar_size]
        # ------------------
        attention_masks = self.exemplar_attm[idx].to(self.device)
        outputs = self.bert(self.exemplar_input[idx].to(self.device), attention_mask=attention_masks.unsqueeze(0))
        enc_outputs = outputs[0]
        bsz, seq_len, hidden_dim = enc_outputs.shape
        spans = self.exemplar_span[idx].to(self.device).unsqueeze(0)
        # pdb.set_trace()

        if opts.task_type == "re" or opts.task_type == "rc":

            span1, span2 = torch.split(spans, 1, dim=1)
            span1 = span1.squeeze(1)
            span2 = span2.squeeze(1)
            span_idx1 = span1.unsqueeze(-1)
            span_idx1 = span_idx1.expand(-1, -1, hidden_dim)
            span_repr1 = torch.gather(enc_outputs, 1, span_idx1)
            features1 = span_repr1.view(1, hidden_dim * 2)

            span_idx2 = span2.unsqueeze(-1)
            span_idx2 = span_idx2.expand(-1, -1, hidden_dim)
            span_repr2 = torch.gather(enc_outputs, 1, span_idx2)
            features2 = span_repr2.view(1, hidden_dim * 2)

            features = torch.cat([features1, features2], dim=-1)
        else:
            span_idx = spans.unsqueeze(-1)
            span_idx = span_idx.expand(-1, -1, hidden_dim)
            span_repr = torch.gather(enc_outputs, 1, span_idx)
            # input feature: (span_bsz, hidden_dim*2)
            features = span_repr.view(1, hidden_dim * 2)

        # span_idx = span_idx.expand(-1, -1, hidden_dim)
        # span_repr = torch.gather(enc_outputs, 1, span_idx)
        # features = span_repr.view(1, hidden_dim * 2)



        # inputs = self.input_map(features, params=self.get_subdict(params, "input_map"))
        example_feature = features
        cls_in = features
        # example_feature = outputs[1]
        # example_feature = torch.cat((outputs[1], inputs), dim=-1)

        # scores = self.classes(inputs, params=self.get_subdict(params, "classes"))

        scores = self.sep_classifier[0](cls_in)
        for i in range(1, TASK_NUM + 1):
            temp_score = self.sep_classifier[i](cls_in)
            scores = torch.cat((scores, temp_score), dim=-1)

        return example_feature, scores
        # exemplar_scores.append(scores)
        # return torch.cat(exemplar_scores, dim=0)

    # def alignment(self, params, task_id, idx):
    #     attention_masks = self.exemplar_attm[idx][task_id].to(self.device)
    #     outputs = self.bert(self.exemplar_input[idx].to(self.device), attention_mask=attention_masks.unsqueeze(0))
    #     curr_feature = outputs[1]
    #     example_old_feature = self.exemplar_cls[idx].to(self.device).unsqueeze(0)
    #
    #     loss_exemplar_feat_distill = (1 - (example_old_feature / example_old_feature.norm(dim=-1,
    #                                                                                       keepdim=True) * curr_feature / curr_feature.norm(
    #         dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
    #
    #     return loss

    def update_exem_feat(self, task_id):
        for i in range(self.exemplar_size):
            attention_masks = self.exemplar_attm[i][task_id].to(self.device)
            outputs = self.bert(self.exemplar_input[i].to(self.device), attention_mask=attention_masks.unsqueeze(0))
            self.exemplar_cls[i] = outputs[1].detach().cpu()

    def score(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def clone_params(self, ):
        return OrderedDict({k: v.clone().detach() for k, v in self.meta_named_parameters()})

    def set_history(self, ):
        self.history = {"params": self.clone_params(), "nslots": self.nslots}
        n = 1

    def set_exemplar(self, dataloader, q: int = 20, params=None, label_sets: Union[List, Set, None] = None,
                     collect_none: bool = False, use_input: bool = False, output_only: bool = False,
                     output: Union[str, None] = None, task_id:int = 0):
        self.eval()
        with torch.no_grad():
            inid = []
            attm = []
            spans = []
            label = []
            ofeat = []
            clsfeat = []
            logit = []
            example_batch = []
            num_batches = len(dataloader)
            test_flag = False
            # for batch in tqdm(dataloader, "collecting exemplar", ncols=128):
            for batch in dataloader:
                batch = batch.to(self.device)
                loss = self.forward(batch, params=params, store=True)
                for i in range(self.outputs["input"].size(0)):
                    inid.append(self.outputs["input"][i])
                    attm.append(self.outputs["attm"][i])
                spans.append(self.outputs["spans"])
                ofeat.append(self.outputs["encoded_features"])
                label.append(self.outputs["label"])
                # clsfeat.append(self.outputs["cls"])
                # logit.append(self.outputs["logit"])
                example_batch.append(batch)
                # TODO: test only
                # if test_flag is False:
                #     test_flag = True
                # else:
                #     break
            # inid = torch.cat(inid, dim=0)
            spans = torch.cat(spans, dim=0)
            # attm = torch.cat(attm, dim=0)
            ofeat = torch.cat(ofeat, dim=0)
            label = torch.cat(label, dim=0)
            # clsfeat = torch.cat(clsfeat, dim=0)
            # logit = torch.cat(logit, dim=0)
            nslots = max(self.nslots, torch.max(label).item() + 1)
            exemplar = {}
            if label_sets is None:
                if collect_none:
                    label_sets = range(nslots)
                else:
                    label_sets = range(1, nslots)
            else:
                if collect_none:
                    if 0 not in label_sets:
                        label_sets = sorted([0] + list(label_sets))
                    else:
                        label_sets = sorted(list(label_sets))
                else:
                    label_sets = sorted([t for t in label_sets if t != 0])
            for i in label_sets:
                idx = (label == i)
                if i == 0:
                    # random sample for none type
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    exemplar[i] = numpy.random.choice(nidx, q, replace=False).tolist()
                    continue
                if torch.any(idx):
                    exemplar[i] = []
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    mfeat = torch.mean(ofeat[idx], dim=0, keepdims=True)
                    if len(nidx) < q:
                        exemplar[i].extend(nidx * (q // len(nidx)) + nidx[:(q % len(nidx))])
                    else:
                        for j in range(q):
                            if j == 0:
                                dfeat = torch.sum((ofeat[nidx] - mfeat) ** 2, dim=1)
                            else:
                                cfeat = ofeat[exemplar[i]].sum(dim=0, keepdims=True)
                                cnum = len(exemplar[i])
                                dfeat = torch.sum((mfeat * (cnum + 1) - ofeat[nidx] - cfeat) ** 2, )
                            tfeat = torch.argmin(dfeat)
                            exemplar[i].append(nidx[tfeat])
                            nidx.pop(tfeat.item())
            exemplar = {i: ([inid[idx] for idx in v], [attm[idx] for idx in v], label[v],
                            spans[v]) for i, v in exemplar.items()}
            # exemplar = {i: (inid[v], attm[v], label[v], spans[v]) for i, v in exemplar.items()}
            exemplar_input = []
            exemplar_attm = []
            exemplar_span = []
            exemplar_labels = []
            # exemplar_cls = []
            # exemplar_logit = []
            for label, pack in exemplar.items():
                exemplar_input.extend(pack[0])
                exemplar_attm.extend(pack[1])
                exemplar_span.append(pack[3])
                exemplar_labels.extend([label] * pack[3].size(0))
                # exemplar_cls.append(pack[5])
                # exemplar_logit.append(pack[6])

            # exemplar_input = torch.cat(exemplar_input, dim=0).cpu()
            # exemplar_attm = torch.cat(exemplar_attm, dim=0).cpu()
            exemplar_span = torch.cat(exemplar_span, dim=0).cpu()
            # exemplar_cls = torch.cat(exemplar_cls, dim=0).cpu()
            # exemplar_logit = torch.cat(exemplar_logit, dim=0).cpu()
            exemplar_labels = torch.LongTensor(exemplar_labels).cpu()

            if not output_only or output is not None:
                if output == "train" or output is None:
                    if self.exemplar_input is None:
                        self.exemplar_input = exemplar_input
                        self.exemplar_attm = exemplar_attm
                        self.exemplar_span = exemplar_span
                        self.exemplar_labels = exemplar_labels
                        # self.exemplar_cls = exemplar_cls
                        # self.exemplar_logit = exemplar_logit
                    else:
                        # update cls feat for previous examples
                        # self.update_exem_feat(task_id)
                        self.exemplar_input.extend(exemplar_input)
                        self.exemplar_attm.extend(exemplar_attm)
                        self.exemplar_span = torch.cat((self.exemplar_span, exemplar_span), dim=0)
                        self.exemplar_labels = torch.cat((self.exemplar_labels, exemplar_labels), dim=0)
                        # self.exemplar_cls = torch.cat((self.exemplar_cls, exemplar_cls), dim=0)
                        # self.exemplar_logit = torch.cat((self.exemplar_logit, exemplar_logit), dim=0)

        exem_idx = list(range(self.exemplar_span.size(0)))
        random.shuffle(exem_idx)
        self.random_exemplar_inx = exem_idx
        self.exemplar_size = len(self.random_exemplar_inx)
        self.update_exemplar_feat(task_id, params)
        return {i: (v[0], v[1], v[2].cpu(), v[3].cpu()) for i, v in exemplar.items()}

    def update_exemplar_feat(self, task_id, params):
        # print("---updating exemplar prompts---")
        exem_scores = []
        exem_feats = []
        with torch.no_grad():
            for i in range(self.exemplar_size):
                exem_feat, exem_score = self.replay_forward(params=params, task_id=task_id, idx=i)
                exem_feats.append(exem_feat.detach().cpu())
                exem_scores.append(exem_score.detach().cpu())
            self.exemplar_feature = torch.cat(exem_feats, dim=0)
            self.exemplar_logit = torch.cat(exem_scores, dim=0)


    def initialize(self, exemplar, ninstances: Dict[int, int], gamma: float = 1.0, tau: float = 1.0, alpha: float = 0.5,
                   params=None):
        self.eval()

        with torch.no_grad():
            weight_norm = torch.norm(self.classes.weight[1:self.nslots], dim=1).mean(dim=0)
            label_inits = []
            label_kt = {}
            for label, feats in exemplar.items():
                exemplar_inputs = self.input_map(feats.to(self.device), params=self.get_subdict(params, "input_map"))
                exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                exemplar_scores = exemplar_scores + self.mask
                exemplar_scores[:, 0] = 0
                exemplar_weights = torch.softmax(exemplar_scores * tau, dim=1)
                normalized_inputs = exemplar_inputs / torch.norm(exemplar_inputs, dim=1, keepdim=True) * weight_norm
                proto = (exemplar_weights[:, :1] * normalized_inputs).mean(dim=0)
                knowledge = torch.matmul(exemplar_weights[:, 1:self.nslots], self.classes.weight[1:self.nslots]).mean(
                    dim=0)
                gate = alpha * math.exp(- ninstances[label] * gamma)
                # gate = 1 / (1 + ninstances[label] * gamma)
                rnd = torch.randn_like(proto) * weight_norm / math.sqrt(self.classes.weight.size(1))
                initvec = proto * gate + knowledge * gate + (1 - gate) * rnd
                label_inits.append((label, initvec.cpu()))
                label_kt[label] = exemplar_weights.mean(dim=0).cpu()
            label_inits.sort(key=lambda t: t[0])
            inits = []
            for i, (label, init) in enumerate(label_inits):
                assert label == self.nslots + i
                inits.append(init)
            inits = torch.stack(inits, dim=0)
            self.outputs["new2old"] = label_kt
        return inits.detach()

    def initialize2(self, exemplar, ninstances: Dict[int, int], gamma: float = 1.0, tau: float = 1.0,
                    alpha: float = 0.5, delta: float = 0.5, params=None):
        self.eval()

        def top_p(probs, p=0.9):
            _val, _idx = torch.sort(probs, descending=True, dim=1)
            top_mask = torch.zeros_like(probs).float() - float("inf")
            for _type in range(probs.size(0)):
                accumulated = 0
                _n = 0
                while accumulated < p or _n <= 1:
                    top_mask[_type, _idx[_type, _n]] = 0
                    accumulated += _val[_type, _n]
                    _n += 1
            return top_mask

        with torch.no_grad():
            weight_norm = torch.norm(self.classes.weight[1:self.nslots], dim=1).mean(dim=0)
            label_inits = []
            label_kt = {}
            for label, feats in exemplar.items():
                exemplar_inputs = self.input_map(feats.to(self.device), params=self.get_subdict(params, "input_map"))
                exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                exemplar_scores = exemplar_scores + self.mask
                exemplar_scores[:, 0] = 0
                top_mask = top_p(torch.softmax(exemplar_scores, dim=1))
                exemplar_scores = exemplar_scores + top_mask
                exemplar_scores[:, 0] = 0
                exemplar_weights = torch.softmax(exemplar_scores * tau, dim=1)
                normalized_inputs = exemplar_inputs / torch.norm(exemplar_inputs, dim=1, keepdim=True) * weight_norm
                proto = delta * (exemplar_weights[:, :1] * normalized_inputs).mean(dim=0)
                kweight = (1 - exemplar_weights[:, :1])
                knowledge = torch.matmul(
                    (1 - delta * exemplar_weights[:, :1]) * (exemplar_weights[:, 1:self.nslots] + 1e-8) / torch.clamp(
                        1 - exemplar_weights[:, :1], 1e-8), self.classes.weight[1:self.nslots]).mean(dim=0)
                gate = alpha * math.exp(- ninstances[label] * gamma)
                rnd = torch.randn_like(proto) * weight_norm / math.sqrt(self.classes.weight.size(1))
                initvec = proto * gate + knowledge * gate + (1 - gate) * rnd
                if torch.any(torch.isnan(initvec)):
                    print(proto, knowledge, rnd, gate, exemplar_weights[:, :1], exemplar_scores[-1, :self.nslots])
                    input()
                label_inits.append((label, initvec.cpu()))
                label_kt[label] = exemplar_weights.mean(dim=0).cpu()
            label_inits.sort(key=lambda t: t[0])
            inits = []
            for i, (label, init) in enumerate(label_inits):
                assert label == self.nslots + i
                inits.append(init)
            inits = torch.stack(inits, dim=0)
            self.outputs["new2old"] = label_kt
        return inits.detach()

    def set(self, features: torch.tensor, ids: Union[int, torch.Tensor, List, None] = None, max_id: int = -1):
        with torch.no_grad():
            if isinstance(ids, (torch.Tensor, list)):
                if torch.any(ids > self.nslots):
                    warnings.warn(
                        "Setting features to new classes. Using 'extend' or 'append' is preferred for new classes")
                self.classes.weight[ids] = features
            elif isinstance(ids, int):
                self.classes.weight[ids] = features
            else:
                if max_id == -1:
                    raise ValueError(f"Need input for either ids or max_id")
                self.classes.weight[:max_id] = features

    def append(self, feature):
        with torch.no_grad():
            self.classes.weight[self.nslots] = feature
            self.nslots += 1

    def extend(self, features):
        with torch.no_grad():
            features = features.to(self.device)
            if len(features.size()) == 1:
                warnings.warn("Extending 1-dim feature vector. Using 'append' instead is preferred.")
                self.append(features)
            else:
                nclasses = features.size(0)
                self.classes.weight[self.nslots:self.nslots + nclasses] = features
                self.nslots += nclasses


class BIC(SepCLSFT):
    def __init__(self,input_dim:int,hidden_dim:int,max_slots:int,init_slots:int,device:Union[torch.device, None]=None, **kwargs)->None:
        super().__init__(input_dim,hidden_dim,max_slots,init_slots,device,**kwargs)
        self.correction_weight = nn.Parameter(torch.ones(1, dtype=torch.float, device=self.device, requires_grad=True))
        self.correction_bias = nn.Parameter(torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=True))
        self.correction_stream = [init_slots]

    def add_stream(self, num_classes):
        self.correction_stream.append(self.correction_stream[-1]+num_classes)

    def forward(self, batch, nslots:int=-1, bias_correction:str="none", exemplar:bool=False, exemplar_distill:bool=False, distill:bool=False, return_loss:bool=True, tau:float=1.0, log_outputs:bool=True, params=None):
        assert bias_correction in ["none", "last", "current"]
        if distill:
            assert bias_correction != "current"
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            features, labels = batch
        else:
            features, labels = batch.features, batch.labels
        inputs = self.input_map(features, params=self.get_subdict(params, "input_map"))
        scores = self.classes(inputs, params=self.get_subdict(params, "classes"))
        if nslots == -1:
            scores += self.mask
            nslots = self.nslots
        else:
            scores += self.idx_mask(max_idx=nslots)
        scores[:, 0] = 0
        if bias_correction == "current":
            assert len(self.correction_stream) >= 2
            scores[:, self.correction_stream[-2]:self.correction_stream[-1]] *= self.correction_weight
            scores[:, self.correction_stream[-2]:self.correction_stream[-1]] += self.correction_bias
        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        else:
            labels = labels
        if log_outputs:
            pred = torch.argmax(scores, dim=1)
            acc = torch.mean((pred == labels).float())
            self.outputs["accuracy"] = acc.item()
            self.outputs["prediction"] = pred.detach().cpu()
            self.outputs["label"] = labels.detach().cpu()
            self.outputs["input_features"] = features.detach().cpu()
            self.outputs["encoded_features"] = inputs.detach().cpu()
        if return_loss:
            labels.masked_fill_(labels >= nslots, 0)
            valid = labels < nslots
            nvalid = torch.sum(valid.float())
            if nvalid == 0:
                loss = 0
            else:
                loss = self.crit(scores[valid], labels[valid])
            if distill and self.history is not None:
                old_scores = self.forward(batch, nslots=self.history["nslots"], return_loss=False, log_outputs=False, params=self.history["params"]).detach()
                if bias_correction == "last":
                    old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] *= self.history['correction_weight']
                    old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] += self.history['correction_bias']
                new_scores = scores[:, :self.history["nslots"]]
                loss_distill = - torch.sum(torch.softmax(old_scores*tau, dim=1) * torch.log_softmax(new_scores*tau, dim=1), dim=1).mean()
                d_weight = self.history["nslots"]
                c_weight = (self.nslots - self.history["nslots"])
                loss = ( d_weight * loss_distill+ c_weight* loss) / (d_weight+c_weight)
            if exemplar and self.exemplar_features is not None:
                if self.exemplar_features.size(0) < 128:
                    exemplar_inputs = self.input_map(self.exemplar_features.to(self.device), params=self.get_subdict(params, "input_map"))
                    exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                else:
                    exemplar_scores = []
                    for _beg in range(0, self.exemplar_features.size(0), 128):
                        _features = self.exemplar_features[_beg:_beg+128, :]
                        _inputs = self.input_map(_features.to(self.device), params=self.get_subdict(params, "input_map"))
                        exemplar_scores.append(self.classes(_inputs, params=self.get_subdict(params, "classes")))
                    exemplar_scores = torch.cat(exemplar_scores, dim=0)
                exemplar_scores[:, 0] = 0.
                loss_exemplar = self.crit(exemplar_scores+self.mask, self.exemplar_labels.to(self.device))
                if exemplar_distill:
                    if self.exemplar_features.size(0) < 128:
                        exemplar_old_inputs = self.input_map(self.exemplar_features.to(self.device), params=self.get_subdict(self.history["params"], "input_map"))
                        exemplar_old_scores = self.classes(exemplar_old_inputs, params=self.get_subdict(self.history["params"], "classes"))
                    else:
                        exemplar_old_scores = []
                        for _beg in range(0, self.exemplar_features.size(0), 128):
                            _features = self.exemplar_features[_beg:_beg+128, :]
                            _inputs = self.input_map(_features.to(self.device), params=self.get_subdict(self.history["params"], "input_map"))
                            exemplar_old_scores.append(self.classes(_inputs, params=self.get_subdict(self.history["params"], "classes")))
                        exemplar_old_scores = torch.cat(exemplar_old_scores, dim=0)
                    exemplar_old_scores[:, 0] = 0.
                    if bias_correction == "last":
                        exemplar_old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] *= self.history['correction_weight']
                        exemplar_old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] += self.history['correction_bias']
                    exemplar_old_scores = exemplar_old_scores[:self.history["nslots"]]
                    loss_exemplar_distill = - torch.sum(torch.softmax(exemplar_old_scores[:self.history["nslots"]]*tau, dim=1) * torch.log_softmax(exemplar_scores[:self.history["nslots"]], dim=1), dim=1).mean()
                    d_weight = self.history["nslots"]
                    c_weight = (self.nslots - self.history["nslots"])
                    loss_exemplar = (d_weight * loss_exemplar_distill+ c_weight* loss_exemplar) / (d_weight+c_weight)
                e_weight = self.exemplar_features.size(0)
                loss = (nvalid * loss + e_weight * loss_exemplar) / (nvalid + e_weight)
                if torch.isnan(loss):
                    print(loss, loss_exemplar)
            return loss
        else:
            return scores[:, :nslots]

    def forward_correction(self, *args, **kwargs):
        '''
        training:
            entropy: normal
            distill:
                old, last
                Fold, Fold * correction_weight + correction_bias,
        '''
        if len(args) >= 3:
            args[2] = "current"
        else:
            kwargs["bias_correction"] = "current"
        return self.forward(*args,**kwargs)

    def set_history(self):
        super().set_history()
        self.history["correction_weight"] = self.correction_weight.item()
        self.history["correction_bias"] = self.correction_bias.item()

    def score(self, *args, **kwargs):
        if len(self.correction_stream) >= 2:
            return self.forward_correction(*args, **kwargs)
        else:
            if len(args) >= 3:
                args[2] = "none"
            else:
                kwargs["bias_correction"] = "none"
            return self.forward(*args, **kwargs)

class ICARL(SepCLSFT):
    def __init__(self,input_dim:int,hidden_dim:int,max_slots:int,init_slots:int,device:Union[torch.device, None]=None, **kwargs)->None:
        super().__init__(input_dim,hidden_dim,max_slots,init_slots,device,**kwargs)
        self.none_feat = None

    def set_none_feat(self, dataloader, params=None):
        self.eval()
        with torch.no_grad():
            ifeat = []; ofeat = []; label = []
            num_batches = len(dataloader)
            # for batch in tqdm(dataloader, "collecting exemplar"):
            for batch in dataloader:
                batch = batch.to(self.device)
                loss = self.forward(batch, params=params)
                ifeat.append(self.outputs["input_features"])
                ofeat.append(self.outputs["encoded_features"])
                label.append(self.outputs["label"])
            ifeat = torch.cat(ifeat, dim=0)
            ofeat = torch.cat(ofeat, dim=0)
            label = torch.cat(label, dim=0)
            nslots = max(self.nslots, torch.max(label).item()+1)
            exemplar = {}
            idx = (label == 0)
            self.none_feat = ofeat[idx].mean(dim=0).cpu()
            return self.none_feat

    def score(self, batch, exemplar=None, params=None):
        if exemplar is None:
            exemplar_labels, exemplar_features = self.exemplar_labels, self.exemplar_features
        else:
            exemplar_labels, exemplar_features = exemplar

        inputs = self.input_map(batch.features, params=self.get_subdict(params, "input_map"))
        scores = []
        scores.append(- torch.sum((inputs - self.none_feat.to(inputs.device).unsqueeze(0))**2, dim=1))
        for i in range(1, self.nslots):
            label_idx = (exemplar_labels == i)
            label_features = exemplar_features[label_idx]
            label_inputs = self.input_map(label_features.to(inputs.device), params=self.get_subdict(params, "input_map")).mean(dim=0, keepdim=True)
            scores.append(- torch.sum((inputs - label_inputs)**2, dim=1))
        scores = torch.stack(scores, dim=0).transpose(0, 1)
        labels = batch.labels
        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        pred = torch.argmax(scores, dim=1)
        acc = torch.mean((pred == labels).float())
        labels.masked_fill_(labels >= self.nslots, 0)
        valid = labels < self.nslots
        nvalid = torch.sum(valid.float())
        if nvalid == 0:
            loss = 0
        else:
            loss = self.crit(scores[valid], labels[valid])
        self.outputs["accuracy"] = acc.item()
        self.outputs["prediction"] = pred.detach().cpu()
        self.outputs["label"] = labels.detach().cpu()
        self.outputs["input_features"] = batch.features.detach().cpu()
        self.outputs["encoded_features"] = inputs.detach().cpu()
        return loss

def test():  # sanity check
    m = SepCLSFT(nhead=8, nlayers=3, hidden_dim=512, input_dim=2048, max_slots=30, init_slots=9,
              device=torch.device("cpu"))


if __name__ == "__main__":
    test()
