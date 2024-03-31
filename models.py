from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from torch.nn.functional import kl_div, softmax, log_softmax
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F


class CustomTrainer(Trainer):
    def __init__(self, model_a, model_b, args, mlp_lr, train_dataset, eval_dataset, **kwargs):
        # model_a is the LLaMA model
        # model_b is the MLP model
        super().__init__(model=model_a, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                         **kwargs)
        self.model_a = model_a
        self.model_b = model_b
        # self.model_a_optimizer = model_a_optimizer

        self.model_b_optimizer = torch.optim.Adam(self.model_b.parameters(), lr=mlp_lr)  # Example
        self.atten = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation.
        """
        # Step 1: Forward pass through model_a (LLaMA)
        outputs_a = self.model_a(**inputs)
        llama_loss = outputs_a.loss

        pooled = outputs_a.get("hidden_states")[1:]
        input_ids = inputs["input_ids"]
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            # sequence_lengths = sequence_lengths
        else:
            sequence_lengths = -1

        # pooled = pooled[torch.arange(args.val_batch_size), sequence_lengths]
        pooled = tuple(
            d[torch.arange(batch_size), sequence_lengths] for d in pooled)  # 32 bsz 4096

        inputs_b = pooled
        kl_loss, atten = self.model_b(inputs_b)
        self.atten = atten

        # Combine losses (example: simple addition, you can weight these as needed)
        # combined_loss = llama_loss + 0.01 * kl_loss  # + kl_loss
        combined_loss = llama_loss + 0.001 * kl_loss  # + kl_loss

        return (combined_loss, outputs_a) if return_outputs else combined_loss

    def training_step(self, model, inputs):
        """
        Perform a training step.
        """
        model.train()
        self.model_b.train()
        inputs = self._prepare_inputs(inputs)

        # Separate optimizer step for model_a and model_b
        for optimizer in [self.optimizer, self.model_b_optimizer]:
            optimizer.zero_grad()

        loss = self.compute_loss(model, inputs)
        loss.backward()

        # Update each model with its optimizer
        self.optimizer.step()
        self.model_b_optimizer.step()

        return loss.detach() / self.args.gradient_accumulation_steps


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        layers = []
        hidden_sizes = [input_size // 2, input_size // 2]
        all_sizes = [input_size, input_size] + hidden_sizes + [input_size, input_size]
        for i in range(len(all_sizes) - 1):
            layers.append(nn.Linear(all_sizes[i], all_sizes[i + 1]))
            if i < len(hidden_sizes):  # Don't add activation after the last layer
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.vae_emb2mu = nn.Linear(input_size, input_size)
        self.vae_emb2logvar = nn.Linear(input_size, input_size)
        self.vae_hidden_attention = nn.Parameter(torch.ones(32))

    def estimate(self, emb, emb2mu, emb2logvar):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        logvar = emb2logvar(emb)
        return mean, logvar

    def kl_div(self, mu_q, log_var_q):
        kl = 0.5 * (log_var_q.exp() + mu_q ** 2 - log_var_q - 1).sum(dim=-1).mean()
        return kl

    def reparameterize(self, mu, log_var):
        batch_size = mu.shape[0]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(batch_size, mu.shape[1]).cuda()
        return mu + std * eps

    def forward(self, x):  # (33 128, _, 4096)
        last_hidden = x[-1]  # 128, 4096
        layer_last = torch.stack(x, dim=0)
        layer_num = len(x)
        mu, log_var = self.estimate(last_hidden, self.vae_emb2mu, self.vae_emb2logvar)
        kl_loss = self.kl_div(mu, log_var)
        z = self.reparameterize(mu, log_var)  ##sample_num, basz, hidden_dim
        rec_hidden = self.layers(z)
        # prefix = np.flip(np.logspace(0, layer_num, layer_num, endpoint=True, base=0.9))

        attn = F.softmax(self.vae_hidden_attention).view(layer_num, 1, 1)
        tgt = layer_last * attn
        tgt = tgt.mean(dim=0)
        rec_loss_fct = MSELoss(reduction='none')
        rec_loss = rec_loss_fct(rec_hidden, tgt).mean()


        return rec_loss + kl_loss, attn
