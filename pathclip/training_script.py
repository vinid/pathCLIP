import argparse
from pathclip.model import CLIPTuner
import pandas as pd

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset",  type=str, required=True)
    parser.add_argument("--validation-dataset", type=str, required=True)
    parser.add_argument("--batch-size", default=32, type=int, required=True)
    parser.add_argument("--num_workers", default=4, type=int, required=True)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--evaluation-steps", type=int, required=True)
    parser.add_argument("--save_directory", required=True)
    parser.add_argument("--weight-decay", required=True, type=float)
    parser.add_argument("--comet-tracking", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = load_args()

    train = pd.read_csv(args.training_dataset)
    valid = pd.read_csv(args.validation_dataset)

    cpt = CLIPTuner(lr=args.learning_rate, weight_decay=args.weight_decay, comet_tracking=args.comet_tracking)

    cpt.tuner(train, valid, save_directory=args.save_directory, batch_size=args.batch_size,
              epochs=args.epochs, evaluation_steps=args.evaluation_steps, num_workers=args.num_workers)
