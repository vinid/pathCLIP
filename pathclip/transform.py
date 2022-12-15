from torchvision.transforms import (
    RandomAffine,
    RandomPerspective,
    RandomAutocontrast,
    RandomEqualize,
    RandomRotation,
    RandomCrop,
    RandomHorizontalFlip
)
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _train_transform(n_px):
    return Compose([
        Resize([512], interpolation=InterpolationMode.BICUBIC),
        RandomCrop([n_px]),
        RandomHorizontalFlip(),
        RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=(-15, 15, -15, 15),
            interpolation=InterpolationMode.BILINEAR,
            fill=127,
        ),
        RandomPerspective(
            distortion_scale=0.3,
            p=0.3,
            interpolation=InterpolationMode.BILINEAR,
            fill=127,
        ),
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
