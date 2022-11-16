from comet_ml import Experiment
from torch import nn
from torch import optim
import clip
import tqdm
import torch
from pathclip.dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader


def image_embedder():
    pass

def text_embedder():
    pass

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

class CLIPTuner:

    def __init__(self, lr=5e-5, weight_decay=0.2, comet_tracking=None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device,
                                                jit=False)  # Must set jit=False for training
        if comet_tracking:
            self.experiment = Experiment(comet_tracking)
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
                    texts = clip.tokenize(list_txt).to(self.device)

                    logits_per_image, logits_per_text = self.model(images, texts)

                    ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                    total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text,
                                                                                                ground_truth)) / 2
                    self.experiment.log_metric("loss", total_loss.item(), step=step)

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
                            texts = clip.tokenize(list_txt).to(self.device)

                            logits_per_image, logits_per_text = self.model(images, texts)

                            ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                            total_loss = (self.loss_img(logits_per_image, ground_truth) +
                                          self.loss_txt(logits_per_text, ground_truth)) / 2

                            self.experiment.log_metric("validation_loss", total_loss.item(), step=step)

                        if total_loss < validation_loss:
                            validation_loss = total_loss
                            torch.save(self.model.state_dict(), f"{save_directory}/trained_bs_{batch_size}_lr_{self.hyper_params['lr']}"
                                                           f"_wd_{self.hyper_params['weight_decay']}.pt")

                pbar.close()
