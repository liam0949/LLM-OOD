from transformers import TrainingArguments, Trainer
from transformers import LlamaForSequenceClassification, LlamaTokenizer,DataCollatorWithPadding, PreTrainedModel
import torch
import numpy as np
import random
from data import load
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


class VImodel(PreTrainedModel):
    def __init__(self, llm, decoder=None):
        super(VImodel, self).__init__()
        self.llm = llm
        self.decoder = decoder

    def estimate(self, emb, emb2mu, emb2logvar):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        logvar = emb2logvar(emb)
        return mean, logvar

    def kl_div(self, mu_q, log_var_q):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        # k = mu_q.size(1)
        # mu_diff = mu_p - mu_q
        # mu_diff_sq = torch.mul(mu_diff, mu_diff)
        # logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        # logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        # fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        # kl_divergence = (fs - k + logdet_std_p - logdet_std_q) * 0.5
        kl = 0.5 * (log_var_q.exp() + mu_q ** 2 - log_var_q - 1).sum(dim=-1).mean()
        return kl

    def reparameterize(self, mu, log_var):
        batch_size = mu.shape[0]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(self.sample_size, batch_size, mu.shape[1]).cuda()
        return mu + std * eps

    def forward(self, batch):
        # loss = loss,
        # logits = pooled_logits,
        # past_key_values = transformer_outputs.past_key_values,
        # hidden_states = transformer_outputs.hidden_states,
        # attentions = transformer_outputs.attentions,
        res = self.llm(**batch)

        if self.decoder is not None:
            pass
            # ce_loss = res.get("loss")
            # hidden_states = res.get("hidden_states")
            # logits = res.get("logits")
            #
            # last_hidden = hidden_states[-1]
            # mu, log_var = self.estimate(sequence_output[:, 0, :], self.vae_emb2mu, self.vae_emb2logvar)
            # kl_loss = self.kl_div(mu, log_var)
            # z = self.reparameterize(mu, log_var)  ##sample_num, basz, hidden_dim
            # if self.kl_annealing == "linear":
            #     beta = min(1.0, epoch * self.beta)
        return res if self.decoder is not None else res


llama_checkpoint = "/home/bossjobai/LLM_Projects/llama/llama-2-7b-chat-hf"
llm = LlamaForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=llama_checkpoint,
    num_labels=task_to_labels["sst2"],
    device_map="auto",
    offload_folder="offload",
    trust_remote_code=True
)
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_checkpoint, add_prefix_space=True)
llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llm.config.pad_token_id = llm.config.eos_token_id
model = VImodel(llm).cuda()


class ViCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("label")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


lr = 1e-4
batch_size = 8
num_epochs = 5

training_args = TrainingArguments(
    output_dir="./llama-lora-token-classification",
    learning_rate=lr,
    lr_scheduler_type="constant",
    warmup_ratio=0.1,
    max_grad_norm=0.3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # report_to="wandb",
    fp16=True,
    # gradient_checkpointing=True,
)


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
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


if __name__ == '__main__':
    # datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']
    datasets = ['sst2']
    # datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k', 'clinc150']
    # datasets = ['clinc150', 'bank', 'rostd']
    benchmarks = ()

    for dataset in datasets:
        if dataset == "sst2":
            train_dataset, dev_dataset, test_dataset = load(dataset, llama_tokenizer, max_seq_length=512,
                                                            is_id=True)
            print("train size " + dataset, len(train_dataset))

        else:
            _, _, ood_dataset = load(dataset, llama_tokenizer, max_seq_length=512)
            benchmarks = (('ood_' + dataset, ood_dataset),) + benchmarks
            print("ood size " + dataset, len(ood_dataset))
    llama_data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)
    train_dataset.to_pandas().info()
    llama_trainer = ViCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # test_dataset=test_dataset,
        data_collator=llama_data_collator,
        compute_metrics=compute_metrics
    )

    # llm.config.use_cache = False

    # do_train = True

    # Launch training and log metrics
    print("Training...")
    llama_trainer.train()
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
