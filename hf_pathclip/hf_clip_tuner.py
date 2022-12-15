"""
Slightly edited version of the original script found here:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/contrastive-image-text/run_clip.py

I have essentially removed a couple of things we didn't need


image_path,caption

"""

import logging
import os
import sys
from augmentation import Transform, collate_fn
from arguments import ModelArguments, DataTrainingArguments
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
import transformers
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")

def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.environ["WANDB_API_KEY"] = data_args.wandb

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clip", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    extension = "csv"
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file

    dataset = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.feature_extractor_name)

    model = AutoModel.from_pretrained(model_args.model_name_or_path)
    config = model.config

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    caption_column = data_args.caption_column
    image_column = data_args.image_column

    # 7. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        config.vision_config.image_size, feature_extractor.image_mean, feature_extractor.image_std
    )
    image_transformations = torch.jit.script(image_transformations)

    # we apply augmentation only on the training set
    image_transformations_augmented = Transform(
        config.vision_config.image_size, feature_extractor.image_mean, feature_extractor.image_std, augment=True
    )
    image_transformations_augmented = torch.jit.script(image_transformations_augmented)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = [caption for caption in examples[caption_column]]
        text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    def transform_images_augment(examples):
        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["pixel_values"] = [image_transformations_augmented(image) for image in images]
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[image_column]:
            try:
                Image.open(image_file)
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.filter(
            filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        train_dataset.set_transform(transform_images_augment)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        eval_dataset = eval_dataset.filter(
            filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
        )
        eval_dataset = eval_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        # Transform images on the fly as doing it on the whole dataset takes too much time.
        eval_dataset.set_transform(transform_images)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
