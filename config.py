import numpy as np
import os
import glob
import argparse


dir = "~/LLM_Projects/llama/llama-2-7b-hf"
save_dir = '~/model_cps/LLM-OOD'

def parse_args(script):
    parser = argparse.ArgumentParser(description='LLM OOD script %s' % (script))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=dir, type=str,
                        help="roberta-large;bert-base-uncased")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--task_name", default="sst2", type=str)

    # parser.add_argument("--batch_size", default=64, type=int)
    # parser.add_argument("--val_batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--mlp_lr", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=float)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--project_name", type=str, default="coling2024_ood")
    parser.add_argument("--freeze", action='store_true', help="freeze the model")
    parser.add_argument("--save_results_path", type=str, default=save_dir,
                        help="the path to save results")

    parser.add_argument("--ib", action="store_true", help="If specified, uses the information bottleneck to reduce\
                            the dimensions.")
    parser.add_argument("--kl_annealing", choices=[None, "linear"], default='linear', type=str)
    parser.add_argument("--deterministic", action="store_true", help="If specified, learns the reduced dimensions\
                            through mlp in a deterministic manner.")




    if script == 'train':

        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--val_batch_size", default=64, type=int)
    elif script == 'test':
        # parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--val_batch_size", default=128, type=int)
        # parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')
    else:
        raise ValueError('Unknown script')

    return parser.parse_args()