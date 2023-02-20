import argparse
from pathclip.model import CLIPTuner
import pandas as pd
import json
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import random

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset",  type=str, required=True)
    parser.add_argument("--validation-dataset", type=str, required=True)
    parser.add_argument("--clip-version", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", default=32, type=int, required=True)
    parser.add_argument("--num_workers", default=4, type=int, required=True)
    parser.add_argument("--seed", default=0, type=int, required=True)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--save_directory", required=True)
    parser.add_argument("--weight-decay", required=True, type=float)
    parser.add_argument("--comet-tracking", required=True)
    parser.add_argument("--comet_tags", nargs="*")
    return parser.parse_args()


if __name__ == "__main__":


    args = load_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


    train = pd.read_csv(args.training_dataset)
    valid = pd.read_csv(args.validation_dataset)

    dataset_name = args.training_dataset.split("/")[-1].split(".")

    evaluation_steps = int((len(train)/args.batch_size)/4)

    cpt = CLIPTuner(lr=args.learning_rate, weight_decay=args.weight_decay, comet_tracking=args.comet_tracking,
                    comet_tags=args.comet_tags, model_type=args.clip_version,
                    saving_directory=args.save_directory, batch_size=args.batch_size,
                    epochs=args.epochs, evaluation_steps=evaluation_steps, dataset_name=dataset_name)

    cpt.tuner(train, valid, num_workers=args.num_workers)



