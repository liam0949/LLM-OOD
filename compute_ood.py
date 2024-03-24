from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss

from sklearn.covariance import EmpiricalCovariance
from evaluation import evaluate_ood
import os
import pandas as pd



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
def detect_ood(model,dev_dataloader, test_dataset):
    class_var, class_mean, norm_bank, all_classes = prepare_ood(dev_dataloader)
    res = []
    keys = ["auroc_IN", "fpr95_IN", "aupr_IN"]
    in_scores = compute_ood(test_dataset, model, class_var, class_mean, norm_bank, all_classes)
    for tag, ood_features in benchmarks:
        out_scores = compute_ood(ood_features, model, class_var, class_mean, norm_bank, all_classes)
        results = evaluate_ood(in_scores, out_scores)
        # print("ood result", results)
        res.append(results)
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

def compute_ood(dataloader, model, class_var, class_mean, norm_bank, all_classes):
    model.eval()
    in_scores = []
    # for batch in dataloader:
    #     model.eval()
    #     batch = {key: value.to(args.device) for key, value in batch.items()}
    #     with torch.no_grad():
    #         ood_keys = model.compute_ood(**batch)
    #         in_scores.append(ood_keys)
    for batch in dataloader:
        batch = {key: value.cuda() for key, value in batch.items()}
        labels = batch['labels']
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.get("logits")
            pooled = outputs.get("hidden_states")[-1]

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
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.get("logits")
            pooled = outputs.get("hidden_states")[-1]
            if bank is None:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
            else:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
                bank = torch.cat([bank, bank], dim=0)
                label_bank = torch.cat([label_bank, label_bank], dim=0)

    norm_bank = F.normalize(bank, dim=-1)
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
    model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    model = model.to("cuda")
    model.eval()
