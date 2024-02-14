import datasets
from datasets import load_dataset
import random
import numpy as np
import csv
import sys
import os

datasets.logging.set_verbosity(datasets.logging.ERROR)

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    '20ng': ("text", None),
    'trec': ("text", None),
    'imdb': ("text", None),
    'wmt16': ("en", None),
    'multi30k': ("text", None),
    'clinc150': ("text", None),
    'bank': ("text", None),
    'rostd': ("text", None)
}


def load(task_name, tokenizer, shot=1000000000, max_seq_length=256, is_id=False):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    print("Loading {}".format(task_name))
    if task_name in ('mnli', 'rte'):
        datasets = load_glue(task_name)
    elif task_name == 'sst2':
        datasets = load_sst2(shot, is_id)
    elif task_name == '20ng':
        datasets = load_20ng(shot, is_id)
    elif task_name == 'trec':
        datasets = load_trec(shot, is_id)
    elif task_name == 'imdb':
        datasets = load_imdb(shot, is_id)
    elif task_name == 'wmt16':
        datasets = load_wmt16()
    elif task_name == 'multi30k':
        datasets = load_multi30k()
    elif task_name == 'clinc150':
        datasets = load_clinc(is_id, shot=shot)
    elif task_name == 'rostd':
        datasets = load_clinc(is_id, shot=shot, data_dir="/home/bossjobai/LLM_Projects/datasets/rostd")
    elif task_name == 'bank':
        datasets = load_uood(is_id, shot=shot)

    def preprocess_function(examples):
        inputs = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key] + " " + examples[sentence2_key],)
        )
        result = tokenizer(*inputs, max_length=max_seq_length, truncation=True)
        result["labels"] = examples["label"] if 'label' in examples else 0
        return result
    print(type(datasets['train']))
    train_dataset = list(map(preprocess_function, datasets['train'])) if 'train' in datasets and is_id else None
    dev_dataset = list(map(preprocess_function, datasets['validation'])) if 'validation' in datasets and is_id else None
    test_dataset = list(map(preprocess_function, datasets['test'])) if 'test' in datasets else None
    return train_dataset, dev_dataset, test_dataset


def load_glue(task):
    datasets = load_dataset("glue", task)
    if task == 'mnli':
        test_dataset = [d for d in datasets['test_matched']] + [d for d in datasets['test_mismatched']]
        datasets['test'] = test_dataset
    return datasets


def load_clinc(is_id, shot=100, data_dir="/home/bossjobai/LLM_Projects/datasets/oos"):
    label_list = get_labels(data_dir)

    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    if is_id:
        train_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "train.tsv")), label_map, label_list)
        dev_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "dev.tsv")), label_map, label_list)
        test_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "test.tsv")), label_map, label_list)
        # train_dataset = select_few_shot(shot, train_dataset, "clinc150")
        # dev_dataset = select_few_shot(shot, dev_dataset, "clinc150")
        # dev_dataset = select_few_shot(shot, dev_dataset, "clinc150")
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    else:
        test_dataset = _get_ood(
            _read_tsv(os.path.join(data_dir, "test.tsv")), ["oos"])
        datasets = {'test': test_dataset}
    return datasets


def load_uood(is_id, shot=100000000, data_dir="/home/bossjobai/LLM_Projects/datasets/banking", known_cls_ratio=0.5, dataname='bank'):
    all_label_list_pos = get_labels(data_dir)
    n_known_cls = round(len(all_label_list_pos) * known_cls_ratio)
    known_label_list = list(
        np.random.choice(np.array(all_label_list_pos), n_known_cls, replace=False))

    ood_labels = list(set(all_label_list_pos) - set(known_label_list))
    label_map = {}
    for i, label in enumerate(known_label_list):
        label_map[label] = i

    if is_id:
        train_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "train.tsv")), label_map, known_label_list)
        dev_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "dev.tsv")), label_map, known_label_list)
        test_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "test.tsv")), label_map, known_label_list)
        # train_dataset = select_few_shot(shot, train_dataset, dataname)
        # dev_dataset = select_few_shot(shot, dev_dataset, dataname)
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    else:
        test_dataset = _get_ood(
            _read_tsv(os.path.join(data_dir, "test.tsv")), ood_labels)
        datasets = {'test': test_dataset}
    return datasets


def load_ROSTD(is_id, shot=100, data_dir="/home/bossjobai/LLM_Projects/datasets/rostd"):
    label_list = get_labels(data_dir)
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    ood_list = ['oos']

    if is_id:
        train_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "train.tsv")), label_map, label_list)
        dev_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "dev.tsv")), label_map, label_list)
        test_dataset = _create_examples(
            _read_tsv(os.path.join(data_dir, "test.tsv")), label_map, label_list)
        train_dataset = select_few_shot(shot, train_dataset, "clinc150")
        dev_dataset = select_few_shot(shot, dev_dataset, "clinc150")
        datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    else:
        test_dataset = _get_ood(
            _read_tsv(os.path.join(data_dir, "test.tsv")), ood_list)
        datasets = {'test': test_dataset}
    return datasets


def load_20ng(shot, is_id):
    all_subsets = (
        '18828_alt.atheism', '18828_comp.graphics', '18828_comp.os.ms-windows.misc', '18828_comp.sys.ibm.pc.hardware',
        '18828_comp.sys.mac.hardware', '18828_comp.windows.x', '18828_misc.forsale', '18828_rec.autos',
        '18828_rec.motorcycles', '18828_rec.sport.baseball', '18828_rec.sport.hockey', '18828_sci.crypt',
        '18828_sci.electronics', '18828_sci.med', '18828_sci.space', '18828_soc.religion.christian',
        '18828_talk.politics.guns', '18828_talk.politics.mideast', '18828_talk.politics.misc',
        '18828_talk.religion.misc')
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for i, subset in enumerate(all_subsets):
        dataset = load_dataset('newsgroup', subset)['train']
        examples = [{'text': d['text'], 'label': i} for d in dataset]
        random.shuffle(examples)
        num_train = int(0.8 * len(examples))
        num_dev = int(0.1 * len(examples))
        train_dataset += examples[:num_train]
        dev_dataset += examples[num_train: num_train + num_dev]
        test_dataset += examples[num_train + num_dev:]
    # if is_id:
    #     train_dataset = select_few_shot(shot, train_dataset, "20ng")
    #     dev_dataset = select_few_shot(shot, dev_dataset, "20ng")
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_trec(shot, is_id):
    datasets = load_dataset('trec')
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    idxs = list(range(len(train_dataset)))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    dev_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} for i in
                   idxs[-num_reserve:]]
    train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} for i in
                     idxs[:-num_reserve]]
    test_dataset = [{'text': d['text'], 'label': d['label-coarse']} for d in test_dataset]
    # if is_id:
    #     train_dataset = select_few_shot(shot, train_dataset, "trec")
    #     dev_dataset = select_few_shot(shot, dev_dataset, "trec")
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_imdb(shot, is_id):
    datasets = load_dataset('imdb')
    train_dataset = datasets['train']
    idxs = list(range(len(train_dataset)))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    dev_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label']} for i in idxs[-num_reserve:]]
    train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label']} for i in
                     idxs[:-num_reserve]]
    test_dataset = datasets['test']
    # if is_id:
    #     train_dataset = select_few_shot(shot, train_dataset, "imdb")
    #     dev_dataset = select_few_shot(shot, dev_dataset, "imdb")
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


def load_wmt16():
    datasets = load_dataset('wmt16', 'de-en')
    test_dataset = [d['translation'] for d in datasets['test']]
    datasets = {'test': test_dataset}
    return datasets


def load_multi30k():
    test_dataset = []
    for file_name in ('./data/multi30k/test_2016_flickr.en', './data/multi30k/test_2017_mscoco.en',
                      './data/multi30k/test_2018_flickr.en'):
        with open(file_name, 'r') as fh:
            for line in fh:
                line = line.strip()
                if len(line) > 0:
                    example = {'text': line, 'label': 0}
                    test_dataset.append(example)
    datasets = {'test': test_dataset}
    return datasets


def load_sst2(shot, is_id):
    def process(file_name):
        examples = []
        with open(file_name, 'r') as fh:
            for line in fh:
                splits = line.split()
                label = splits[0]
                text = " ".join(splits[1:])
                examples.append(
                    {'sentence': text, 'label': int(label)}
                )
        return examples

    datasets = load_dataset('glue', 'sst2')
    train_dataset = datasets['train']
    dev_dataset = datasets['validation']
    test_dataset = process('./data/sst2/test.data')
    # if is_id:
    #     train_dataset = select_few_shot(shot, train_dataset, "sst2")
    #     dev_dataset = select_few_shot(shot, dev_dataset, "sst2")
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test': test_dataset}
    return datasets


# train_dataset = [{'text': train_dataset[i]['text'], 'label': train_dataset[i]['label-coarse']} for i in
#                      idxs[:-num_reserve]]
def select_few_shot(shot, trainset, task_name):
    # examples = []
    few_examples = []
    sentence1_key, sentence2_key = task_to_keys[task_name]
    from collections import defaultdict
    sorted_examples = defaultdict(list)

    for example in trainset:
        # if example.label in self.known_label_list and np.random.uniform(0, 1) <= args.labeled_ratio:
        #     examples.append(example)
        sorted_examples[example["label"]] = sorted_examples[example["label"]] + [example[sentence1_key]]
    for k, v in sorted_examples.items():
        arr = np.array(v)
        np.random.shuffle(arr)
        for elems in arr[:shot]:
            few_examples.append({sentence1_key: elems, 'label': k})

    return few_examples


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def _create_examples(lines, label_map, know_labels):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if len(line) != 2:
            continue
        # guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        if label in know_labels:
            examples.append(
                {'text': text_a, 'label': label_map[label]})
    return examples


def _get_ood(lines, ood_labels):
    out_examples = []

    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        if len(line) != 2:
            continue
        # guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        if label in ood_labels:
            out_examples.append(
                {'text': text_a, 'label': 0})

    return out_examples


def get_labels(data_dir):
    """See base class."""
    import pandas as pd
    test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
    labels = np.unique(np.array(test['label']))

    return labels
