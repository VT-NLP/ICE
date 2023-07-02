import json
from typing import Union, List
def get_label2id(streams:Union[List[int], None]=None, stream_file:Union[str, None]=None):
    if streams is None:
        with open(stream_file, "rt") as sf:
            streams = json.load(sf)
    label2id = {0: 0}
    for task in streams:
        for label in task:
            if label not in label2id:
                label2id[label] = len(label2id)
    return label2id


def get_task_stat(dataset, perm_id):
    if dataset == "MAVEN" or dataset == "ACE":
        PERM = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2], [0]]
    elif dataset == "FewNERD":
        PERM = [[0, 1, 2, 3 ,4 ,5 ,6 ,7], [7, 6, 5, 4, 3, 2, 1, 0], [0, 4, 5, 2, 7 ,3 ,6 ,1], [5, 3, 6, 0, 2, 1, 4, 7], [1, 7, 6, 5, 3, 4, 2, 0], [0]]
    elif dataset == "TACRED":
        PERM = [[0, 1, 2, 3 ,4 ,5 ,6 ,7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [2, 6, 8, 1, 9, 4, 5, 0, 7, 3], [7, 3, 1, 5, 8, 6, 4, 0, 9, 2], [2, 7, 3, 8, 0, 4, 6, 9, 5, 1], [0]]

    if dataset == "MAVEN":
        init_task_event_num = [33, 30, 39, 35, 31]
    elif dataset == "ACE":
        init_task_event_num = [9, 6, 5, 5, 8]
    elif dataset == "FewNERD":
        # init_task_event_num = [6, 8, 6, 7, 10, 12, 8, 9]
        init_task_event_num = [12, 8, 3, 11, 6, 9, 13, 4]
    elif dataset == "TACRED":
        init_task_event_num = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4 ]

    if dataset == "MAVEN":
        if perm_id == 5:
            task_event_num = [168]
        else:
            task_event_num = [init_task_event_num[i] for i in PERM[perm_id]]
    elif dataset == "ACE":
        if perm_id == 5:
            task_event_num = [33]
        else:
            task_event_num = [init_task_event_num[i] for i in PERM[perm_id]]

    elif dataset == "FewNERD":
        if perm_id == 5:
            task_event_num = [66]
        else:
            task_event_num = [init_task_event_num[i] for i in PERM[perm_id]]
    elif dataset == "TACRED":
        if perm_id == 5:
            task_event_num = [40]
        else:
            task_event_num = [init_task_event_num[i] for i in PERM[perm_id]]


    na_task_event_num = [1] + task_event_num
    acc_num = [1]
    for i in range(len(task_event_num)):
        acc_num.append(acc_num[i] + task_event_num[i])

    return PERM, len(task_event_num), task_event_num, na_task_event_num, acc_num



