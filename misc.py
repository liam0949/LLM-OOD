# Load Llama 2 Tokenizer
from transformers import AutoTokenizer, DataCollatorWithPadding
llama_tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint, add_prefix_space=True)
llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
llama_tokenizer.pad_token = llama_tokenizer.eos_token

def llama_preprocessing_function(examples):
    return llama_tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

llama_tokenized_datasets = data.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
llama_tokenized_datasets = llama_tokenized_datasets.rename_column("target", "label")
llama_tokenized_datasets.set_format("torch")

# Data collator for padding a batch of examples to the maximum length seen in the batch
llama_data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)

from transformers import AutoModelForSequenceClassification
import torch
llama_model =  AutoModelForSequenceClassification.from_pretrained(
  pretrained_model_name_or_path=llama_checkpoint,
  num_labels=2,
  device_map="auto",
  offload_folder="offload",
  trust_remote_code=True
)
llama_model.config.pad_token_id = llama_model.config.eos_token_id

from peft import get_peft_model, LoraConfig, TaskType
llama_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

llama_model = get_peft_model(llama_model, llama_peft_config)
llama_model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer

llama_model = llama_model.cuda()

lr = 1e-4
batch_size = 8
num_epochs = 5
training_args = TrainingArguments(
    output_dir="llama-lora-token-classification",
    learning_rate=lr,
    lr_scheduler_type= "constant",
    warmup_ratio= 0.1,
    max_grad_norm= 0.3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    fp16=True,
    gradient_checkpointing=True,
)


from transformers import Trainer

class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weights, pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

llama_trainer = WeightedCELossTrainer(
    model=llama_model,
    args=training_args,
    train_dataset=llama_tokenized_datasets['train'],
    eval_dataset=llama_tokenized_datasets["val"],
    data_collator=llama_data_collator,
    compute_metrics=compute_metrics
)


import evaluation
import numpy as np
def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


class VImodel(torch.nn.Module):
    def __init__(self, llm, decoder = None):
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
        kl = 0.5*(log_var_q.exp() + mu_q ** 2 - log_var_q - 1).sum(dim=-1).mean()
        return kl
    def reparameterize(self, mu, log_var):
        batch_size = mu.shape[0]
        std = torch.exp(0.5*log_var)
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
            ce_loss = res.get("loss")
            hidden_states = res.get("hidden_states")
            logits = res.get("logits")

            last_hidden = hidden_states[-1]
            mu, log_var = self.estimate(sequence_output[:, 0, :], self.vae_emb2mu, self.vae_emb2logvar)
            kl_loss = self.kl_div(mu, log_var)
            z = self.reparameterize(mu, log_var)  ##sample_num, basz, hidden_dim
            if self.kl_annealing == "linear":
                beta = min(1.0, epoch * self.beta)
        return res if self.decoder is not None else (res, ce_loss)





