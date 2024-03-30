from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from data import load
from sklearn.covariance import EmpiricalCovariance
from evaluation import evaluate_ood
from torch.utils.data import DataLoader
import os
import pandas as pd
from utils import find_subdir_with_smallest_number
from config import parse_args
import evaluate

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key].append(i[key])
        new_dict[key] = np.mean(np.array(new_dict[key]))
    return new_dict


# outputs["auroc_IN"] = auroc_in
# outputs["fpr95_IN"] = fpr_95_in
# outputs["aupr_IN"] = aupr_in
def detect_ood(model, dev_dataloader, test_dataset, benchmarks, data_collator):
    class_var, class_mean, norm_bank, all_classes = prepare_ood(model,dev_dataloader)
    print("finshed prepare")
    res = []
    keys = ["auroc_IN", "fpr95_IN", "aupr_IN"]

    in_scores = compute_ood(test_dataset, model, class_var, class_mean, norm_bank, all_classes)
    # print("in_scores",len(in_scores))

    for tag, ood_features in benchmarks:
        dataloader = DataLoader(ood_features, batch_size=4, collate_fn=data_collator)
        out_scores = compute_ood(dataloader, model, class_var, class_mean, norm_bank, all_classes)
        print(tag,"out score finshed")
        results = evaluate_ood(in_scores, out_scores)
        # print("ood result", results)
        res.append(results)
        del dataloader, ood_features

        # wandb.log(results, step=num_steps)
    res = merge_keys(res, keys)
    return res


def save_results(args, test_results):
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    var = [args.task_name, args.seed, args.ib]
    names = ['dataset', 'seed', 'is_ib']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = args.task_name + '_results.csv'
    results_path = os.path.join(args.save_results_path, file_name)

    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results_path, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])
        df1 = df1.append(new, ignore_index=True)
        df1.to_csv(results_path, index=False)
    data_diagram = pd.read_csv(results_path)

    print('test_results')
    print(data_diagram)


def compute_ood( dataloader, model, class_var, class_mean, norm_bank, all_classes):
    model.eval()
    in_scores = []
    # dataloader = DataLoader(dev_dataset, batch_size=128, collate_fn=data_collator)
    for batch in dataloader:
        batch = {key: value.cuda() for key, value in batch.items()}
        input_ids = batch['input_ids']
        batch['labels'] = None
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.get("logits")
            pooled = outputs.get("hidden_states")[-1]

            if input_ids is not None:
                batch_size = input_ids.shape[0]
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                # sequence_lengths = sequence_lengths
            else:
                sequence_lengths = -1

            # pooled = pooled[torch.arange(args.val_batch_size), sequence_lengths]
            pooled = pooled[torch.arange(batch_size,device=pooled.device), sequence_lengths]


        ood_keys = None
        softmax_score = F.softmax(logits, dim=-1).max(-1)[0]

        maha_score = []
        for c in all_classes:
            centered_pooled = pooled - class_mean[c].unsqueeze(0)
            ms = torch.diag(centered_pooled @ class_var @ centered_pooled.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1)
        maha_score = maha_score.min(-1)[0]
        maha_score = -maha_score

        norm_pooled = F.normalize(pooled, dim=-1)
        cosine_score = norm_pooled @ norm_bank.t()
        cosine_score = cosine_score.max(-1)[0]

        energy_score = torch.logsumexp(logits, dim=-1)

        ood_keys = {
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist(),
        }
        in_scores.append(ood_keys)
    return in_scores


# from torch.utils.data import DataLoader
#
# train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
# metric = evaluate.load("accuracy")
# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)
#
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])
#
# metric.compute()
def prepare_ood(model, dataloader=None):
    bank = None
    label_bank = None
    model.eval()
    for batch in dataloader:
        batch = {key: value.cuda() for key, value in batch.items()}
        labels = batch['labels']
        input_ids = batch['input_ids']
        batch['labels'] = None
        with torch.no_grad():
            outputs = model(**batch)
            # logits = outputs.get("logits")
            pooled = outputs.get("hidden_states")[-1]

            if input_ids is not None:
                batch_size = input_ids.shape[0]
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                # sequence_lengths = sequence_lengths
            else:
                sequence_lengths = -1

            pooled = pooled[torch.arange(batch_size), sequence_lengths]
            # pooled = pooled[:, sequence_lengths]
            # print("poled:", pooled.shape)

            if bank is None:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
            else:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
                bank = torch.cat([bank, bank], dim=0)
                label_bank = torch.cat([label_bank, label_bank], dim=0)

    norm_bank = F.normalize(bank, dim=-1).cuda()
    N, d = bank.size()
    all_classes = list(set(label_bank.tolist()))
    class_mean = torch.zeros(max(all_classes) + 1, d).cuda()
    for c in all_classes:
        class_mean[c] = (bank[label_bank == c].mean(0))
    centered_bank = (bank - class_mean[label_bank]).detach().cpu().numpy()
    precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
    class_var = torch.from_numpy(precision).float().cuda()
    return class_var, class_mean, norm_bank, all_classes


if __name__ == '__main__':
    args = parse_args("test")
    args.ib = False
    out_dir = os.path.join(args.save_results_path, args.task_name, str(args.seed), str(args.ib))
    print(out_dir)
    out_dir = find_subdir_with_smallest_number(out_dir)
    print(out_dir)
    assert out_dir is not None

    model = AutoPeftModelForSequenceClassification.from_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side="right"
    model = model.to("cuda")
    model.config.pad_token_id = model.config.eos_token_id
    model.config.output_hidden_states = True

    # print(model.config.output_hidden_states)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    print(tokenizer.padding_side)

    _, dev_dataset, test_dataset = load(args.task_name, tokenizer, max_seq_length=args.max_seq_length,
                                        is_id=True)
    dev_dataset.to_pandas().info()
    test_dataset.to_pandas().info()
    ood_datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']
    # ood_datasets = ['rte', 'sst2', ]

    benchmarks = ()

    if args.task_name in ["sst2", "imdb"]:
        ood_datasets = list(set(ood_datasets) - set(["sst2", "imdb"]))
    else:
        ood_datasets = list(set(ood_datasets) - set([args.task_name]))
    for dataset in ood_datasets:
        _, _, ood_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
        benchmarks = (('ood_' + dataset, ood_dataset),) + benchmarks
        ood_dataset.to_pandas().info()

    # train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    # outputs.hidden_states[-1]
    # to
    # match
    # outputs.last_hidden_states
    # exactly
    ##test acc
    test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size, collate_fn=data_collator)
    # print("",len(test_dataloader))
    eval_dataloader = DataLoader(dev_dataset, batch_size=args.val_batch_size, collate_fn=data_collator)
    metric = evaluate.load("accuracy")

    model.eval()
    for batch in test_dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        # print(batch["input_ids"][0])
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        # # hs = outputs.hidden_states  # 33 128, 66, 4096
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    # print("test acc:", metric.compute())

    test_acc = metric.compute()

    metric = evaluate.load("accuracy")
    for batch in eval_dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        # print(batch["input_ids"][0])
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        # # hs = outputs.hidden_states  # 33 128, 66, 4096
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    eval_acc = metric.compute()
    ood_res = detect_ood(model, eval_dataloader, test_dataloader, benchmarks, data_collator)
    final_res = dict({"test_acc": test_acc, 'eval_acc': eval_acc}, **ood_res)

    save_results(args, final_res)
    # print(res)
    ## comput OOD
