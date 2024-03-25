from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from torch.nn.functional import kl_div, softmax, log_softmax


class CustomTrainer(Trainer):
    def __init__(self, model_a, model_b, args, train_dataset, eval_dataset, tokenizer, model_a_optimizer,
                 model_b_optimizer, **kwargs):
        # model_a is the LLaMA model
        # model_b is the MLP model
        super().__init__(model=model_a, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                         tokenizer=tokenizer, **kwargs)
        self.model_a = model_a
        self.model_b = model_b
        self.model_a_optimizer = model_a_optimizer
        self.model_b_optimizer = model_b_optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation.
        """
        # Step 1: Forward pass through model_a (LLaMA)
        outputs_a = self.model_a(**inputs)
        llama_loss = outputs_a.loss

        # Step 2: Extract decoder hidden states and use them as input to model_b (MLP)
        hidden_states = outputs_a.decoder_hidden_states[-1]  # Assuming we take the last layer's hidden states
        inputs_b = hidden_states.detach()  # Detach to prevent gradients flowing into LLaMA during MLP's backward pass
        predictions = self.model_b(inputs_b)

        # Step 3: Compute custom loss (combination of LLaMA loss and MLP loss)
        # Assuming we have ground truth labels for both models' outputs
        labels = inputs["labels"]
        cross_entropy_loss = nn.CrossEntropyLoss()(predictions, labels)

        # KL-divergence loss part (assuming we have some target distribution `target_distrib`)
        # target_distrib = ... (define or obtain your target distribution)
        # kl_loss = kl_div(log_softmax(predictions, dim=1), target_distrib, reduction='batchmean')

        # Combine losses (example: simple addition, you can weight these as needed)
        combined_loss = llama_loss + cross_entropy_loss  # + kl_loss

        return (combined_loss, outputs_a) if return_outputs else combined_loss

    def training_step(self, model, inputs):
        """
        Perform a training step.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Separate optimizer step for model_a and model_b
        for optimizer in [self.model_a_optimizer, self.model_b_optimizer]:
            optimizer.zero_grad()

        loss = self.compute_loss(model, inputs)
        loss.backward()

        # Update each model with its optimizer
        self.model_a_optimizer.step()
        self.model_b_optimizer.step()

        return loss.item()
from transformers import AutoModelForSequenceClassification, AdamW

model_a = AutoModelForSequenceClassification.from_pretrained("path_to_llama")
model_b = MyMLPModel(...)  # Define your MLP model

model_a_optimizer = AdamW(model_a.parameters(), lr=1e-5)
model_b_optimizer = AdamW(model_b.parameters(), lr=1e-4)  # Assuming a different learning rate for illustration

training_args = TrainingArguments(
    output_dir="./model_a_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = CustomTrainer(
    model_a=model_a,
    model_b=model_b,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    model_a_optimizer=model_a_optimizer,
    model_b_optimizer=model_b_optimizer
)
# Start the training process
trainer.train()

# Save the LLaMA model to the specified output directory
trainer.save_model()


##split
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        all_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(all_sizes) - 1):
            layers.append(nn.Linear(all_sizes[i], all_sizes[i + 1]))
            if i < len(hidden_sizes):  # Don't add activation after the last layer
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput


class CustomTrainer(Trainer):
    def __init__(self, model, mlp_model, args, train_dataset=None, eval_dataset=None, tokenizer=None, model_init=None,
                 compute_metrics=None):
        super().__init__(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer,
                         model_init=model_init, compute_metrics=compute_metrics)
        self.mlp_model = mlp_model
        # Initialize separate optimizer for MLP if needed
        self.mlp_optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=0.001)  # Example

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the combined loss.
        """
        outputs = model(**inputs)
        # Assume decoder_hidden_states is part of outputs for demonstration
        decoder_hidden_states = outputs.hidden_states[-1]  # Taking the last layer's hidden state

        # Forward pass through MLP
        mlp_outputs = self.mlp_model(decoder_hidden_states)

        # Example targets for demonstration
        mlp_targets = inputs.get("mlp_targets")  # You need to include this in your dataset

        # Calculate LLaMA model loss (e.g., cross-entropy)
        llama_loss = outputs.loss

        # Calculate MLP model loss (e.g., KL divergence)
        mlp_loss = nn.KLDivLoss()(mlp_outputs, mlp_targets)

        # Combine losses (Assuming equal weighting for simplicity)
        combined_loss = llama_loss + mlp_loss

        if return_outputs:
            return combined_loss, outputs
        return combined_loss

    def training_step(self, model, inputs):
        """
        Perform a training step with separate optimizers.
        """
        model.train()
        self.mlp_model.train()

        inputs = self._prepare_inputs(inputs)

        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss.backward()

        self.optimizer.step()
        self.mlp_optimizer.step()
        self.optimizer.zero_grad()
        self.mlp_optimizer.zero_grad()

        return loss.detach()
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

llama_model = ... # Load your LLaMA model
mlp_model = MLP(input_size=..., hidden_sizes=[128, 64], output_size=...) # Adjust sizes as needed

trainer = CustomTrainer(
    model=llama_model,
    mlp_model=mlp_model,
    args=training_args,
    train_dataset=train_dataset,  #
    eval_dataset=eval_dataset,  # Assuming you have an eval_dataset
    tokenizer=tokenizer,  # Your tokenizer
)

trainer.train()

