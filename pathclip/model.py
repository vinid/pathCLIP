from comet_ml import Experiment
from torch import nn
from torch import optim
import clip
import tqdm
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
from torchvision.transforms import (
    RandomAffine,
    RandomPerspective,
    RandomAutocontrast,
    RandomEqualize,
    RandomRotation
)
from pathclip.dataset import *
from torch.utils.data import DataLoader
from PIL import Image
from datetime import datetime

from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _train_transform(n_px):
    return Compose([

        RandomPerspective(
            distortion_scale=0.3,
            p=0.3,
            interpolation=InterpolationMode.BILINEAR,
            fill=127,
        ),
        RandomRotation(degrees=(0, 180)),
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])




def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def zero_shot_classification(model, preprocess, images, labels, device, num_workers=1):
    image_embeddings = image_embedder(model, preprocess, images, device, num_workers)
    text_embeddings = text_embedder(model, labels, device, num_workers)

    score = image_embeddings.dot(text_embeddings.T)
    predictions = [labels[np.argmax(i)] for i in score]

    return predictions


def image_embedder(model, preprocess, list_of_images, device="cuda", num_workers=1):
    batch_size = 64
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

def text_embedder(model, list_of_labels, device="cuda", num_workers=1):
    batch_size = 64
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

    def __init__(self, lr=5e-5, weight_decay=0.2, comet_tracking=None, px_size=224):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device,
                                                jit=False)  # Must set jit=False for training
        self.train_preprocess = _train_transform(px_size)
        if comet_tracking:
            self.experiment = Experiment(comet_tracking, project_name="pathclip")
        else:
            self.experiment = Experiment()

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

    def tuner(self, train_dataframe, validation_dataframe, save_directory, batch_size=4, epochs=5,
              evaluation_steps=500, num_workers=1):

        start_time = str(datetime.now())
        train_dataset = ImageCaptioningDataset(train_dataframe, self.preprocess)
        validation_dataset = ImageCaptioningDataset(validation_dataframe, self.preprocess)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)

        validation_loss = 10000
        step = 0
        with self.experiment.train():

            for epoch in range(epochs):
                pbar = tqdm.tqdm(position=0, total=len(train_dataloader))
                pbar.set_description(f"{epoch}/{epochs}")

                for batch in train_dataloader:

                    self.optimizer.zero_grad()

                    list_image, list_txt = batch

                    images = list_image
                    images = images.to(self.device)
                    texts = clip.tokenize(list_txt, truncate=True).to(self.device)

                    logits_per_image, logits_per_text = self.model(images, texts)

                    logits_per_image = 20*logits_per_image
                    logits_per_text = 20*logits_per_text

                    ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                    total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text,
                                                                                                ground_truth)) / 2
                    step = step + 1

                    total_loss.backward()
                    if self.device == "cpu":
                        self.optimizer.step()
                    else:
                        convert_models_to_fp32(self.model)
                        self.optimizer.step()
                        clip.model.convert_weights(self.model)
                    pbar.update(1)

                    if step % evaluation_steps == 0:

                        for batch in validation_dataloader:
                            pbar.set_description("Currently Validating")

                            with torch.no_grad():

                                list_image, list_txt = batch

                                images = list_image
                                images = images.to(self.device)
                                texts = clip.tokenize(list_txt, truncate=True).to(self.device)

                                logits_per_image, logits_per_text = self.model(images, texts)

                                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                                total_loss = (self.loss_img(logits_per_image, ground_truth) +
                                              self.loss_txt(logits_per_text, ground_truth)) / 2

                                self.experiment.log_metric("validation_loss", total_loss.item(), step=step)

                            if total_loss < validation_loss:
                                validation_loss = total_loss
                                torch.save(self.model.state_dict(), f"{save_directory}/trained_bs_{batch_size}_lr_{self.hyper_params['lr']}"
                                                               f"_wd_{self.hyper_params['weight_decay']}"
                                                                    f"_{start_time}_{self.experiment.get_name()}.pt")

                pbar.close()
