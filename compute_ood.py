from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model = model.to("cuda")
model.eval()


def compute_ood(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
):
    outputs = roberta(
        input_ids,
        attention_mask=attention_mask,
    )
    sequence_output = outputs[0]
    logits, pooled = classifier(sequence_output)

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
    return ood_keys


def prepare_ood(model,dataloader=None):
    bank = None
    label_bank = None
    for batch in dataloader:
        batch = {key: value.cuda() for key, value in batch.items()}
        labels = batch['labels']
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        sequence_output = outputs[0]
        logits, pooled = classifier(sequence_output)
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


def detect_ood():
    model.prepare_ood(dev_dataloader)
    res = {}
    for tag, ood_features in benchmarks:
        results = evaluate_ood(args, model, test_dataset, ood_features, tag=tag)
        # print("ood result", results)
        res = dict(res, **results)
        wandb.log(results, step=num_steps)
    return res
