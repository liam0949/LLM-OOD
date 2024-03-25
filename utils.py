import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    outputs = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "labels": labels,
    }
    return outputs

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0]


import os

# Mock base directory for demonstration purposes
# base_dir = "/mnt/data/example_base_dir"
#
# # Create mock directories to simulate the Linux filesystem structure for this example
# os.makedirs(f"{base_dir}/checkpoints-600", exist_ok=True)
# os.makedirs(f"{base_dir}/checkpoints-700", exist_ok=True)
# os.makedirs(f"{base_dir}/checkpoints-500", exist_ok=True)


def find_subdir_with_smallest_number(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    filtered_subdirs = [d for d in subdirs if d.startswith("checkpoints-")]

    if not filtered_subdirs:
        return None  # No sub directories found

    # Extract numbers and find the subdir with the smallest number
    min_number = float('inf')
    min_subdir = None
    for subdir in filtered_subdirs:
        try:
            number = int(subdir.split("-")[-1])
            if number < min_number:
                min_number = number
                min_subdir = subdir
        except ValueError:
            # Skip subdirs that do not end with a number
            continue

    if min_subdir is not None:
        return os.path.abspath(os.path.join(base_dir, min_subdir))
    else:
        return None  # No valid subdir found


# Find the requested subdir
# result = find_subdir_with_smallest_number(base_dir)
# result


