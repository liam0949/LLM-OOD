from transformers import AutoTokenizer, DataCollatorWithPadding
import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, AutoConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn, AverageMeter, accuracy
from datasets import load_metric
from evaluation import evaluate_ood
import wandb
import warnings

from data import load
import random
from datasets import load_metric

task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20ng': 20,
    'trec': 6,
    'clinc150': 150,
    "bank": round(77 * 0.5),
    'rostd': 3
    # "clincMix": 150
}

task_to_metric = {
    'sst2': 'sst2',
    'imdb': 'sst2',
    '20ng': 'mnli',
    'trec': 'mnli',
    'clinc150': 'mnli',
    'bank': 'mnli',
    'rostd': 'mnli',
    # 'clincMix': 'mnli',
}


class ViCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# training_args = TrainingArguments(
#     output_dir="./llama-lora-token-classification",
#     learning_rate=lr,
#     lr_scheduler_type="constant",
#     warmup_ratio=0.1,
#     max_grad_norm=0.3,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=num_epochs,
#     weight_decay=0.001,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     # report_to="wandb",
#     fp16=True,
#     # gradient_checkpointing=True,
# )


def compute_metrics(eval_pred):
    metric_name = task_to_metric["sst2"]
    metric = load_metric("glue", metric_name)

    logits, labels = eval_pred  # eval_pred is the tuple of predictions and labels returned by the model

    preds = np.argmax(logits, axis=1)
    result = metric.compute(predictions=preds, references=labels)
    if len(result) > 1:
        result["score"] = np.mean(list(result.values())).item()
    accuracy = result["accuracy"]
    return {'accuracy': accuracy}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks):
    pass


class ViCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", default="EleutherAI/gpt-j-6B", type=str,
    parser.add_argument("--model_name_or_path", default="distilbert/distilbert-base-uncased", type=str,
                        help="roberta-large;bert-base-uncased")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--task_name", default="sst2", type=str)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--val_batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--learning_rate_vae", default=1e-3, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--project_name", type=str, default="coling2024_ood")
    parser.add_argument("--shot", type=int, default=100)
    parser.add_argument("--freeze", action='store_true', help="freeze the model")
    parser.add_argument("--save_results_path", type=str, default='/data2/liming/LLM-OOD',
                        help="the path to save results")

    parser.add_argument("--ib", action="store_true", help="If specified, uses the information bottleneck to reduce\
                        the dimensions.")
    parser.add_argument("--kl_annealing", choices=[None, "linear"], default='linear', type=str)
    parser.add_argument("--deterministic", action="store_true", help="If specified, learns the reduced dimensions\
                        through mlp in a deterministic manner.")

    parser.add_argument("--beta", type=float, default=1e-3, help="Defines the weight for the information bottleneck\
                        loss.")
    parser.add_argument("--sample_size", type=int, default=20, help="Defines the number of samples for the ib method.")

    args = parser.parse_args()
    set_seed(args)

    wandb.init(project=args.project_name)
    wan_config = wandb.config
    wan_config.learning_rate = args.learning_rate
    wan_config.task_name = args.task_name

    ##load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    num_labels = task_to_labels[args.task_name]
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     args.model_name_or_path,
    #     num_labels=num_labels,
    #     revision="float16",
    #     torch_dtype=torch.float16
    # )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels
    )
    model.config.output_hidden_states = True

    # model.print_trainable_parameters()
    model = model.cuda()



    # datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']
    # datasets = ['sst2', 'imdb', 'trec', '20ng']
    # datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k', 'clinc150']
    # datasets = ['clinc150', 'bank', 'rostd']
    benchmarks = ()
    train_dataset, dev_dataset, test_dataset = load(args.task_name, tokenizer, max_seq_length=args.max_seq_length,
                                                    is_id=True)
    print("train size " + args.task_name, len(train_dataset))
    train_dataset.to_pandas().info()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_results_path,args.task_name),
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        # gradient_checkpointing=True,
    )

    llama_trainer = ViCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # test_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # llm.config.use_cache = False

    # do_train = True

    # Launch training and log metrics
    print("Training...")
    llama_trainer.train()

    print("Testing...")
    llama_trainer.evaluate(test_dataset)
    # if do_train:
    #     train_result = trainer.train()
    #     metrics = train_result.metrics
    #     trainer.log_metrics("train", metrics)
    #     trainer.save_metrics("train", metrics)
    #     trainer.save_state()
    #     print(metrics)
    #
    # # Save model
    # print("Saving last checkpoint of the model...")
    # os.makedirs(output_dir, exist_ok=True)
    # trainer.model.save_pretrained(output_dir)
    #
    # # Free memory for merging weights
    # del model
    # del trainer
    # torch.cuda.empty_cache()