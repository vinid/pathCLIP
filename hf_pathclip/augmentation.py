import torch
from torchvision.transforms import (
    RandomAffine,
    RandomPerspective,
    RandomAutocontrast,
    RandomEqualize,
)
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode


class Transform(torch.nn.Module):

    def __init__(self, image_size, mean, std, augment=False):
        """
        :param image_size:
        :param mean:
        :param std:
        :param augment: True => applies augmentation. Useful to differentiate training and validation
        """
        super().__init__()
        if augment:
            self.transforms = torch.nn.Sequential(
                Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                ConvertImageDtype(torch.float),
                Normalize(mean, std),
            )
        else:
            self.transforms = torch.nn.Sequential(
                Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                ConvertImageDtype(torch.float),
                Normalize(mean, std))

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }
