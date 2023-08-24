from typing import Any, List, Tuple, Union, Callable
from transformers import BertTokenizerFast
import json
import os
from tqdm import tqdm
import torch
import numpy as np
import transformers
import pdb

# data_root = "./ACE"
dataset = "TACRED"


train_file = f"./data/{dataset}/{dataset}.train.jsonl"
dev_file = f"./data/{dataset}/{dataset}.dev.jsonl"
test_file = f"./data/{dataset}/{dataset}.test.jsonl"
feature_root = f"./data/features/{dataset}/"

bt = BertTokenizerFast.from_pretrained("bert-large-cased")
device_name = "cuda:0"
model = transformers.BertModel.from_pretrained("bert-large-cased").to(torch.device(device_name))

model.eval()
max_length=512
expand_context=False


collected = set()
_sentence_buffer = []


def clear_sentences():
    with torch.no_grad():

        sentences = [t[0] for t in _sentence_buffer]
        sentence_ids = [t[1] for t in _sentence_buffer]
        length = [len(t) for t in sentences]

        masks = torch.FloatTensor([[1] * len(t) for t in sentences])
        sentences = torch.LongTensor([t for t in sentences])

        outputs = model(input_ids=sentences.to(torch.device(device_name)),
                             attention_mask=masks.to(torch.device(device_name)))
        for i, (s_id, s_l) in enumerate(zip(sentence_ids, length)):
            feature_path = os.path.join(feature_root, s_id)
            features = outputs[0][i]

            features = features.cpu().numpy()
            np.save(file=feature_path, arr=features)
    _sentence_buffer.clear()

def add_sentence(sentence_id, piece_ids):
    collected.add(sentence_id)
    feature_path = os.path.join(feature_root, sentence_id)
    if not os.path.exists(f"{feature_path}.npy"):
        _sentence_buffer.append((piece_ids, sentence_id))
        if len(_sentence_buffer) >= 1:
            clear_sentences()



def process_file(file_path):
    instances = []
    with open(file_path, "rt") as fp:
        for document_line in tqdm(fp):
            document = json.loads(document_line)
            sent_id = document["sentence_id"]
            piece_ids = document["piece_ids"]
            if len(piece_ids) >= 511 or len(piece_ids) <= 2:
                continue
            add_sentence(sent_id, piece_ids)
            document["feature_path"] = f"{sent_id}"
            instances.append(document)

    return instances

train_instances = process_file(train_file)
dev_instances = process_file(dev_file)
test_instances = process_file(test_file)
# print(train_instances)

with open(f"/data/{dataset}/{dataset}.train.feat.jsonl", "wt") as fp:
    for instance in train_instances:
        fp.write(json.dumps(instance) + "\n")
with open(f"/data/{dataset}/{dataset}.dev.feat.jsonl", "wt") as fp:
    for instance in dev_instances:
        fp.write(json.dumps(instance) + "\n")
with open(f"/data/{dataset}/{dataset}.test.feat.jsonl", "wt") as fp:
    for instance in test_instances:
        fp.write(json.dumps(instance) + "\n")
