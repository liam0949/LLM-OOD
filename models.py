from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from torch.nn.functional import kl_div, softmax, log_softmax
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss


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
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            # sequence_lengths = sequence_lengths
        else:
            sequence_lengths = -1

        # pooled = pooled[torch.arange(args.val_batch_size), sequence_lengths]
        pooled = tuple(
            d[torch.arange(batch_size), sequence_lengths] for d in pooled)  # 32 bsz 4096
        # Step 2: Extract decoder hidden states and use them as input to model_b (MLP)
        # Assuming we take the last layer's hidden states
        inputs_b = pooled  # Detach to prevent gradients flowing into LLaMA during MLP's backward pass
        kl_loss, _ = self.model_b(inputs_b)

        # Step 3: Compute custom loss (combination of LLaMA loss and MLP loss)
        # Assuming we have ground truth labels for both models' outputs
        # labels = inputs["labels"]
        # cross_entropy_loss = nn.CrossEntropyLoss()(predictions, labels)

        # KL-divergence loss part (assuming we have some target distribution `target_distrib`)
        # target_distrib = ... (define or obtain your target distribution)
        # kl_loss = kl_div(log_softmax(predictions, dim=1), target_distrib, reduction='batchmean')

        # Combine losses (example: simple addition, you can weight these as needed)
        combined_loss = llama_loss + kl_loss  # + kl_loss

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
        self.model_a_optimizer.step()
        self.model_b_optimizer.step()

        return loss.item()


# from transformers import AutoModelForSequenceClassification, AdamW
#
# model_a = AutoModelForSequenceClassification.from_pretrained("path_to_llama")
# model_b = MyMLPModel(...)  # Define your MLP model
#
# model_a_optimizer = AdamW(model_a.parameters(), lr=1e-5)
# model_b_optimizer = AdamW(model_b.parameters(), lr=1e-4)  # Assuming a different learning rate for illustration
#
# training_args = TrainingArguments(
#     output_dir="./model_a_output",
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     logging_dir='./logs',
#     logging_steps=10,
# )
#
# trainer = CustomTrainer(
#     model_a=model_a,
#     model_b=model_b,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     model_a_optimizer=model_a_optimizer,
#     model_b_optimizer=model_b_optimizer
# )
# # Start the training process
# trainer.train()
#
# # Save the LLaMA model to the specified output directory
# trainer.save_model()


##split


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        layers = []
        hidden_sizes = [input_size * 2, input_size * 4, input_size * 2]
        all_sizes = [input_size] + hidden_sizes + [input_size]
        for i in range(len(all_sizes) - 1):
            layers.append(nn.Linear(all_sizes[i], all_sizes[i + 1]))
            if i < len(hidden_sizes):  # Don't add activation after the last layer
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.vae_emb2mu = nn.Linear(input_size, input_size)
        self.vae_emb2logvar = nn.Linear(input_size, input_size)
        self.vae_hidden_attention = nn.Parameter(torch.ones(24))

    def estimate(self, emb, emb2mu, emb2logvar):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        logvar = emb2logvar(emb)
        return mean, logvar

    def kl_div(self, mu_q, log_var_q):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""

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
        rec_loss = rec_loss_fct(rec_hidden, tgt).sum(dim=-1).mean()

        # print(rec_hidden.size())
        # print()
        # print(torch.stack(hidden_states[0:10], dim=0).size())
        # prefix = np.flip(np.logspace(0, 22, 23, endpoint=True, base=0.9))
        # tgt = torch.stack(tuple(p1*h1 for p1, h1 in zip(prefix, hidden_states[:23])), dim=0)[:,:,0,:].mean(dim=0).unsqueeze(dim=1)
        # tgt = hidden_states[11][:, 0].unsqueeze(dim=1)
        # tgt = torch.stack(hidden_states, dim=0)[:, :, 0, :].max(dim=0)[0]
        # attn = F.softmax(self.vae_hidden_attention).view(24, 1, 1)
        # tgt = torch.stack(hidden_states, dim=0)[:, :, 0, :] * attn
        # # print(tgt.size())
        # tgt = tgt.sum(dim=0).unsqueeze(dim=1)
        # rec_loss_fct = MSELoss(reduction='none')
        # rec_loss = rec_loss_fct(rec_hidden, tgt).sum(dim=-1).mean()
        # # print(ce_loss)
        # # loss["loss"] = ce_loss + (beta if self.kl_annealing == "linear" else self.beta) * (kl_loss + rec_loss)
        # loss["loss"] = ce_loss + self.beta * (kl_loss + rec_loss)

        return rec_loss + kl_loss, attn


from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

# class CustomTrainer(Trainer):
#     def __init__(self, model, mlp_model, args, train_dataset=None, eval_dataset=None, tokenizer=None, model_init=None,
#                  compute_metrics=None):
#         super().__init__(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer,
#                          model_init=model_init, compute_metrics=compute_metrics)
#         self.mlp_model = mlp_model
#         # Initialize separate optimizer for MLP if needed
#         self.mlp_optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=args.mlp_lr)  # Example
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         Compute the combined loss.
#         """
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         pooled = outputs.get("hidden_states")[1:]
#         input_ids = inputs["input_ids"]
#         if input_ids is not None:
#             batch_size = input_ids.shape[0]
#             # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
#             sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
#             sequence_lengths = sequence_lengths % input_ids.shape[-1]
#             # sequence_lengths = sequence_lengths
#         else:
#             sequence_lengths = -1
#
#         # pooled = pooled[torch.arange(args.val_batch_size), sequence_lengths]
#         pooled = tuple(d[torch.arange(batch_size, device=pooled.device), sequence_lengths] for d in pooled) #32 bsz 4096
#         # Assume decoder_hidden_states is part of outputs for demonstration
#         decoder_hidden_states = outputs.hidden_states[-1]  # Taking the last layer's hidden state
#
#         # Forward pass through MLP
#         mlp_outputs = self.mlp_model(decoder_hidden_states)
#
#         # Example targets for demonstration
#         mlp_targets = inputs.get("mlp_targets")  # You need to include this in your dataset
#
#         # Calculate LLaMA model loss (e.g., cross-entropy)
#         llama_loss = outputs.loss
#
#         # Calculate MLP model loss (e.g., KL divergence)
#         mlp_loss = nn.KLDivLoss()(mlp_outputs, mlp_targets)
#
#         # Combine losses (Assuming equal weighting for simplicity)
#         combined_loss = llama_loss + mlp_loss
#
#         if return_outputs:
#             return combined_loss, outputs
#         return combined_loss
#
#     def training_step(self, model, inputs):
#         """
#         Perform a training step with separate optimizers.
#         """
#         model.train()
#         self.mlp_model.train()
#
#         inputs = self._prepare_inputs(inputs)
#
#         loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
#         loss.backward()
#
#         self.optimizer.step()
#         self.mlp_optimizer.step()
#         self.optimizer.zero_grad()
#         self.mlp_optimizer.zero_grad()
#
#         return loss.detach()
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
# )
#
# llama_model = ... # Load your LLaMA model
# mlp_model = MLP(input_size=..., hidden_sizes=[128, 64], output_size=...) # Adjust sizes as needed
#
# trainer = CustomTrainer(
#     model=llama_model,
#     mlp_model=mlp_model,
#     args=training_args,
#     train_dataset=train_dataset,  #
#     eval_dataset=eval_dataset,  # Assuming you have an eval_dataset
#     tokenizer=tokenizer,  # Your tokenizer
# )
