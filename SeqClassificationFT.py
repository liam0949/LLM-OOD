
import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn, AverageMeter, accuracy
from datasets import load_metric
from model import RobertaForSequenceClassification, BertForSequenceClassification, Dreconstruction
from evaluation import evaluate_ood
import wandb
import warnings
from data import load

warnings.filterwarnings("ignore")

task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20ng': 20,
    'trec': 6,
    'clinc150': 150,
    "bank": round(77 * 0.75) + 1,
    # "clincMix": 150
}

task_to_metric = {
    'sst2': 'sst2',
    'imdb': 'sst2',
    '20ng': 'mnli',
    'trec': 'mnli',
    'clinc150': 'mnli',
    'bank': 'mnli',
    # 'clincMix': 'mnli',
}


def train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True,
                                  drop_last=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                             num_training_steps=total_steps)

    def detect_ood():
        model.prepare_ood(dev_dataloader)
        res = {}
        for tag, ood_features in benchmarks:
            results = evaluate_ood(args, model, test_dataset, ood_features, tag=tag)
            # print("ood result", results)
            res = dict(res, **results)
            wandb.log(results, step=num_steps)
        return res

    # num_steps_pre = 0
    # num_steps_rec = 0
    # loss_avg = AverageMeter()
    # eval_loss_avg = AverageMeter()
    # if args.train_rec:
    # for epoch in range(int(10)):
    #     model.zero_grad()
    #     model.train()
    #     # epochs += 1
    #     for step, batch in enumerate(tqdm(train_dataloader)):
    #         batch = {key: value.to(args.device) for key, value in batch.items()}
    #         # if args.train_rec:
    #         #     batch["insampler"] = rector
    #         outputs = model(**batch)
    #         loss, _ = outputs[0], outputs[1]
    #         loss.backward()
    #         num_steps_pre += 1
    #         optimizer.step()
    #         scheduler.step()
    #         model.zero_grad()
    #         wandb.log({'pre_loss': loss.item()}, step=num_steps_pre)
    #
    #         # loss_avg.update(loss.item(), len(batch['labels']))

    # rector = Dreconstruction(model.config.hidden_size, args.rec_num)
    # rector.cuda()
    # optimizer_mlp = torch.optim.Adam(rector.parameters(), lr=1e-4)
    # schedulor_rec = torch.optim.lr_scheduler.ExponentialLR(optimizer_mlp, 0.7)
    # rector.rec = False
    # best_eval = float('inf')
    # patient = 0
    # model.eval()

    # epoches = 0
    # for epoch in range(int(50)):
    #     rector.zero_grad()
    #
    #     for step, batch in enumerate(tqdm(train_dataloader)):
    #         rector.train()
    #
    #         # batch = {key: value.to(args.device) for key, value in batch.items()}
    #         batch = {key: value.cuda() for key, value in batch.items()}
    #         # labels = batch['labels']
    #         outputs = model.bert(
    #             input_ids=batch['input_ids'],
    #             attention_mask=batch['attention_mask'],
    #         )
    #         pooled = outputs[1]
    #         loss = rector(pooled, dropout=args.rec_drop) * 10000
    #         loss.backward()
    #         loss_avg.update(loss.item(), n=len(batch['labels']))
    #         num_steps_rec += 1
    #         optimizer_mlp.step()
    #         # scheduler.step()
    #         model.zero_grad()
    #     # eval
    #     schedulor_rec.step()
    #
    #     for step, batch in enumerate(tqdm(dev_dataloader)):
    #         rector.eval()
    #         # batch = {key: value.to(args.device) for key, value in batch.items()}
    #         batch = {key: value.cuda() for key, value in batch.items()}
    #         # labels = batch['labels']
    #         outputs = model.bert(
    #             input_ids=batch['input_ids'],
    #             attention_mask=batch['attention_mask'],
    #         )
    #         pooled = outputs[1]
    #         val_loss = rector(pooled, dropout=args.rec_drop) * 10000
    #         eval_loss_avg.update(val_loss.item(), n=len(batch['labels']))
    #     print("\n")
    #     print("train rec loss:", loss_avg.avg)
    #     print("val rec loss:", eval_loss_avg.avg)
    #     wandb.log({'train_loss_rec': loss_avg.avg}, step=num_steps_rec)
    #     wandb.log({'val_loss_rec': eval_loss_avg.avg}, step=num_steps_rec)
    #     if eval_loss_avg.avg < best_eval:
    #         best_eval = eval_loss_avg.avg
    #         patient = 0
    #     else:
    #         patient += 1
    #
    #     loss_avg.reset()
    #     eval_loss_avg.reset()
    #     if patient > 5:
    #         break
    #
    # rector.rec = True

    best_eval = -float('inf')
    eval_fre = 5
    # epochs = 0
    patient = 0
    loss_avg = AverageMeter()
    loss_avg_mse = AverageMeter()

    loss_avg.reset()
    loss_avg_mse.reset()
    num_steps = 0
    final_res = {}

    # eval_loss_avg.reset()
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        model.train()
        # epochs += 1
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            # if args.train_rec:
            #     batch["insampler"] = rector
            batch["epoch"] = epoch + 1
            batch["mode"] = "train"
            outputs = model(**batch)
            loss, logits = outputs[0], outputs[1]
            logits = logits
            labels = outputs[2]
            acc = accuracy(logits, labels)
            loss['loss'].backward()
            num_steps += 1
            optimizer.step()
            # scheduler.step()
            model.zero_grad()
            wandb.log({'loss': loss['loss'].item()}, step=num_steps)
            wandb.log({'loss_mse': loss['mse'].item()}, step=num_steps)
            wandb.log({'train acc': acc}, step=num_steps)
            # wandb.log({'std': model.std_p_in.item()}, step=num_steps)
            # wandb.log({'mean': model.mu_p_in.item()}, step=num_steps)

            loss_avg.update(loss['loss'].item(), len(batch['labels']))
            loss_avg_mse.update(loss['mse'].item(), len(batch['labels']))

            # wandb.log({'cos_loss': cos_loss.item()}, step=num_steps)
        print("\n")
        print("train loss of epoch:", epoch,loss_avg.avg)
        results = evaluate(args, model, dev_dataset, tag="dev")
        print("dev result:", results["dev_accuracy"])
        wandb.log(results, step=num_steps)
        # print(results)
        dev_res = results["dev_accuracy"]

        loss_avg.reset()
        # eval_loss_avg.reset()
        if dev_res > best_eval:
            results = evaluate(args, model, test_dataset, tag="test")
            print("test result:", results['test_accuracy'])
            wandb.log(results, step=num_steps)
            #ood_res = detect_ood()
            best_eval = dev_res
            # final_res = dict(ood_res, **{"test_acc": results['test_accuracy'], 'eval_acc': best_eval})
            final_res = dict({"test_acc": results['test_accuracy'], 'eval_acc': best_eval})
            patient = 0
        else:
            patient += 1

        # if patient > 100 and epoch > 20:
        #     save_results(args, final_res)
        #     break


from sklearn.metrics import f1_score


def evaluate(args, model, eval_dataset, tag="train"):
    metric_name = task_to_metric[args.task_name]
    metric = load_metric("glue", metric_name)

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()
        return result

    dataloader = DataLoader(eval_dataset, batch_size=args.val_batch_size, collate_fn=collate_fn)

    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        # labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        batch["mode"] = tag
        outputs = model(**batch)
        logits = outputs[1].detach().cpu().numpy()
        labels = outputs[2].detach().cpu().numpy()

        label_list.append(labels)
        logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    # if tag =='test':
    #     results = compute_metrics(preds, labels)
    #     results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    # else:
    #     f1_scores = f1_score(labels, np.argmax(preds, -1), average='macro')
    #     results = f1_scores
    #     results = {tag + "_accuracy": results}
    results = compute_metrics(preds, labels)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}

    return results


import os
import pandas as pd


def save_results(args, test_results):
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    var = [args.task_name, args.seed, args.rec_drop, args.train_rec, args.shot]
    names = ['dataset', 'seed', 'rec_drop', "is_rec", 'shot']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = 'results_rec.csv'
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

    # print('test_results')
    # print(data_diagram)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, help="roberta-large;")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--task_name", default="sst2", type=str)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--val_batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="ood")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss", type=str, default="margin")
    parser.add_argument("--shot", type=int, default=100)
    parser.add_argument("--train_rec", action='store_true', help="save test results")
    parser.add_argument("--convex", action='store_true', help="save test results")
    parser.add_argument("--freeze", action='store_true', help="freeze the model")
    parser.add_argument("--rec_num", type=int, default=10)
    parser.add_argument("--rec_drop", type=float, default=30)
    parser.add_argument("--save_results_path", type=str, default='/data2/liming/few-shot-nlp',
                        help="the path to save results")

    parser.add_argument("--ib", action="store_true", help="If specified, uses the information bottleneck to reduce\
                    the dimensions.")
    parser.add_argument("--activation", type=str, choices=["tanh", "sigmoid", "relu"],
                        default="relu")
    parser.add_argument("--deterministic", action="store_true", help="If specified, learns the reduced dimensions\
                    through mlp in a deterministic manner.")
    parser.add_argument("--vid", action="store_true", help="If specified, learns the reduced dimensions\
                    through mlp in a deterministic manner.")
    parser.add_argument("--uvid", action="store_true", help="If specified, learns the reduced dimensions\
                    through mlp in a deterministic manner.")
    parser.add_argument("--svid", action="store_true", help="If specified, learns the reduced dimensions\
                    through mlp in a deterministic manner.")
    parser.add_argument("--beta", type=float, default=1e-1, help="Defines the weight for the information bottleneck\
                    loss.")
    parser.add_argument("--beta", type=float, default=1e-1, help="Defines the weight for the information bottleneck\
                        loss.")

    args = parser.parse_args()

    # wandb.init(project=args.project_name, name=args.task_name + '-' + str(args.alpha) + "_" + args.loss)
    wandb.init(project="few-shot-ood-nlp",
               name=args.task_name + '-' + 'rec_' + str(args.train_rec) + '-' + 'freeze_' + str(args.freeze),
               entity="li-ming")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    num_labels = task_to_labels[args.task_name]
    if args.model_name_or_path.startswith('roberta'):
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        # config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        config.train_rec = args.train_rec
        config.rec_num = args.rec_num
        config.freeze = args.freeze
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config,
        )
        model.to(0)
    elif args.model_name_or_path.startswith('bert'):
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        # config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        config.freeze = args.freeze
        config.train_rec = args.train_rec
        config.rec_num = args.rec_num
        config.rec_drop = args.rec_drop
        config.beta = args.beta
        config.convex = args.convex
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config,
        )
        model.to(0)

    # datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k', 'clinc150']
    datasets = ['clinc150', 'bank']
    benchmarks = ()

    for dataset in datasets:
        if dataset == args.task_name:
            train_dataset, dev_dataset, test_dataset = load(dataset, tokenizer, shot=args.shot,
                                                            max_seq_length=args.max_seq_length, is_id=True)
            _, _, ood_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
            benchmarks = (('ood_' + dataset, ood_dataset),) + benchmarks
        # else:
        #     _, _, ood_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
        #     benchmarks = (('ood_' + dataset, ood_dataset),) + benchmarks

        # if dataset == 'clinc150':
        #     _, _, ood_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
        #     benchmarks = (('ood_' + dataset, ood_dataset),) + benchmarks
    train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks)


if __name__ == "__main__":
    main()
