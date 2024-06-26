from transformers import AutoTokenizer, DataCollatorWithPadding
import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, AutoConfig, \
    LlamaTokenizer, LlamaForSequenceClassification
from transformers import BitsAndBytesConfig
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn, AverageMeter, accuracy
from datasets import load_metric
from evaluation import evaluate_ood
import wandb
import warnings
import os
import datetime
import evaluate
from data import load
import random
from datasets import load_metric
from config import parse_args
from models import CustomTrainer, MLP
import warnings

warnings.filterwarnings("ignore")
from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",

    ]
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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

def compute_metrics(eval_pred):
    metric_name = task_to_metric["sst2"]
    metric = evaluate.load("glue", metric_name)

    logits, labels = eval_pred.predictions, eval_pred.label_ids  # eval_pred is the tuple of predictions and labels returned by the model


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




if __name__ == '__main__':

    args = parse_args("train")
    set_seed(args)

    wandb.init(project=args.project_name, name=args.task_name+"VI" + str(datetime.datetime.now()))
    wan_config = wandb.config
    wan_config.learning_rate = args.learning_rate
    # wan_config.task_name = args.task_name

    ##load model
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    num_labels = task_to_labels[args.task_name]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        device_map="auto",
        offload_folder="offload",
        # load_in_8bit=True,
        # quantization_config=BitsAndBytesConfig(
        #     # load_in_4bit=model_args.bits == 4,
        #     load_in_8bit=True
        #     # llm_int8_threshold=6.0,
        #     # llm_int8_has_fp16_weight=False
        #     # bnb_4bit_compute_dtype=compute_dtype,
        #     # bnb_4bit_use_double_quant=model_args.double_quant,
        #     # bnb_4bit_quant_type=model_args.quant_type,
        # ),
        trust_remote_code=True,
    )
    model.config.pad_token_id = model.config.eos_token_id
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     args.model_name_or_path,
    #     num_labels=num_labels,
    #     trust_remote_code = True
    # )
    model.config.output_hidden_states = True
    # model.config.keys_to_ignore_at_inference = ["hidden_states"]
    model.config.keys_to_ignore_at_inference = ["past_key_values","hidden_states"]

    # model.print_trainable_parameters()
    # model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.cuda()
    mlp = MLP(model.config.hidden_size).cuda()

    benchmarks = ()
    train_dataset, dev_dataset, test_dataset = load(args.task_name, tokenizer, max_seq_length=args.max_seq_length,
                                                    is_id=True)
    print("train size " + args.task_name, len(train_dataset))
    train_dataset.to_pandas().info()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args.ib = True
    out_dir = os.path.join(args.save_results_path, args.task_name, str(args.seed),str(args.ib))
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        # max_grad_norm=0.3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        # save_strategy = "epoch",
        save_steps=500,
        load_best_model_at_end=True,
        # metric_for_best_model="accuracy",
        # greater_is_better=True,
        save_total_limit=2,
        report_to="wandb",
        bf16=True
        # gradient_checkpointing=True,
    )

    llama_trainer = CustomTrainer(
        model_a=model,
        model_b= mlp,
        mlp_lr=args.mlp_lr,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # test_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # llm.config.use_cache = False

    # do_train = True

    print("vanilla performance...")
    test_acc = llama_trainer.evaluate(test_dataset)
    wandb.log({"test_acc": test_acc})
    # wandb.log({"atten": wandb.Histogram(llama_trainer.atten.view(32,).cpu().numpy())})
    print(llama_trainer.atten.view(32,).detach().cpu().numpy())

    # Launch training and log metrics
    print("Training...")
    llama_trainer.train()
    # wandb.log({"atten": wandb.Histogram(llama_trainer.atten.view(32,).cpu().numpy())})
    print(llama_trainer.atten.view(32,).detach().cpu().numpy())

    print("Testing...")
    test_acc = llama_trainer.evaluate(test_dataset)
    wandb.log({"test_acc": test_acc})
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
