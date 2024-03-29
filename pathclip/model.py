from comet_ml import Experiment
from torch import nn
from torch import optim
import clip
import tqdm
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import math
from torch.optim.lr_scheduler import LinearLR

from pathclip.dataset import *
from pathclip.transform import _train_transform
from pathclip.scheduler import cosine_lr
from torch.utils.data import DataLoader
from PIL import Image
from datetime import datetime
import json

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def zero_shot_classification(model, preprocess, images, labels, device, num_workers=1, batch_size=32):
    image_embeddings = image_embedder(model, preprocess, images, device, num_workers, batch_size)
    text_embeddings = text_embedder(model, labels, device, num_workers, batch_size)

    score = image_embeddings.dot(text_embeddings.T)
    predictions = [labels[np.argmax(i)] for i in score]

    return predictions


def image_embedder(model, preprocess, list_of_images, device="cuda", num_workers=1, batch_size=32):
    train_dataset = ImageDataset(list_of_images, preprocess)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    image_embeddings = []

    total = len(list_of_images) // batch_size
    pbar = tqdm.tqdm(total=total, position=0)
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            image_embeddings.extend(model.encode_image(images).detach().cpu().numpy())

            pbar.update(1)
        pbar.close()

    image_embeddings = np.array(image_embeddings)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    return image_embeddings

def text_embedder(model, list_of_labels, device="cuda", num_workers=1, batch_size=32):
    train_dataset = CaptioningDataset(list_of_labels)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    text_embeddings = []
    total = len(list_of_labels) // batch_size

    pbar = tqdm.tqdm(total=total, position=0)
    with torch.no_grad():
        for captions in dataloader:
            idx = clip.tokenize(captions, truncate=True).to(device)
            text_embeddings.extend(model.encode_text(idx).detach().cpu().numpy())

            pbar.update(1)

        pbar.close()

    text_embeddings = np.array(text_embeddings)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    return text_embeddings

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

class CLIPTuner:

    def __init__(self, model_type="ViT-B/32", lr=5e-5, weight_decay=0.1, warmup=50,
                 comet_tracking=None, px_size=224, comet_tags=None, batch_size=128,
                 saving_directory="", evaluation_steps= 100, epochs=10, dataset_name=""):

        self.save_directory = saving_directory
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.evaluation_steps = evaluation_steps

        start_time = str(datetime.now())

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load(model_type, device=self.device,
                                                jit=False)  # Must set jit=False for training
        self.warmup = warmup
        self.train_preprocess = _train_transform(px_size)
        if comet_tracking:
            self.experiment = Experiment(comet_tracking, project_name="pathclip")
        else:
            self.experiment = Experiment()


        saving_args = {
            "bs": batch_size,
            "comet_tags": comet_tags,
            "weight_decay": weight_decay,
            "learning_rate": lr,
            "total_epochs": epochs,
            "dataset_name": dataset_name,
            "evaluation_steps": evaluation_steps
        }

        additional_name = ""
        for key, value in saving_args.items():
            additional_name += f"{key}_{value}_"

        self.model_name = f"{additional_name}_{start_time}_{self.experiment.get_name()}.pt"

        with open(f"{saving_directory}/{self.model_name}_config.json", "w") as filino:
            filino.write(json.dumps(saving_args))

        if comet_tags:
            self.experiment.add_tags(comet_tags)

        if self.device == "cpu":
            self.model.float()
        else:
            clip.model.convert_weights(self.model)

        self.hyper_params = {
            "lr": lr,
            "weight_decay": weight_decay
        }

        self.experiment.log_parameters(self.hyper_params)

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(self.model.parameters(),
                                    lr=self.hyper_params["lr"],
                                    weight_decay=self.hyper_params["weight_decay"])

    def tuner(self, train_dataframe, validation_dataframe, num_workers=1):

        start_time = str(datetime.now())
        train_dataset = ImageCaptioningDataset(train_dataframe, self.train_preprocess)
        validation_dataset = ImageCaptioningDataset(validation_dataframe, self.preprocess)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=num_workers)
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, num_workers=num_workers)
        num_batches_per_epoch = len(train_dataloader)
        total_steps = len(train_dataloader) * self.epochs

        scheduler = cosine_lr(self.optimizer, self.hyper_params["lr"], self.warmup, total_steps)

        with self.experiment.train():

            for epoch in range(self.epochs):
                pbar = tqdm.tqdm(position=0, total=len(train_dataloader))
                pbar.set_description(f"{epoch}/{self.epochs}")

                for i, batch in enumerate(train_dataloader):
                    self.optimizer.zero_grad()
                    step = num_batches_per_epoch * epoch + i
                    scheduler(step)

                    list_image, list_txt = batch

                    images = list_image
                    images = images.to(self.device)
                    texts = clip.tokenize(list_txt, truncate=True).to(self.device)

                    logits_per_image, logits_per_text = self.model(images, texts)

                    logit_scale = self.model.logit_scale.exp()
                    self.experiment.log_metric("logit_scale", logit_scale.item(), step=step)

                    logits_per_image = logits_per_image
                    logits_per_text = logits_per_text

                    ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                    total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text,
                                                                                                ground_truth)) / 2
                    total_loss.backward()
                    new_lr = scheduler(step)
                    self.experiment.log_metric("learning_rate", new_lr, step=step)

                    if self.device == "cpu":
                        self.optimizer.step()
                    else:
                        convert_models_to_fp32(self.model)
                        self.optimizer.step()
                        clip.model.convert_weights(self.model)

                    pbar.update(1)

                    with torch.no_grad():
                        unwrap_model(self.model).logit_scale.clamp_(0, math.log(100))

                    if step % self.evaluation_steps == 0:
                        torch.save(self.model.state_dict(), f"{self.save_directory}/"
                                                            f"_{self.model_name}"
                                                            "_steps_%06d.pt" % step)

                        for batch in validation_dataloader:
                            pbar.set_description("Currently Validating")

                            with torch.no_grad():

                                list_image, list_txt = batch

                                images = list_image
                                images = images.to(self.device)
                                texts = clip.tokenize(list_txt, truncate=True).to(self.device)

                                logits_per_image, logits_per_text = self.model(images, texts)

                                logits_per_image = logits_per_image
                                logits_per_text = logits_per_text

                                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                                total_loss = (self.loss_img(logits_per_image, ground_truth) +
                                              self.loss_txt(logits_per_text, ground_truth)) / 2

                                self.experiment.log_metric("validation_loss", total_loss.item(), step=step)

                        pbar.set_description(f"{epoch}/{self.epochs}")

                pbar.close()

