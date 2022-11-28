from torchvision.transforms import (
    RandomAffine,
    RandomPerspective,
    RandomAutocontrast,
    RandomEqualize,
    RandomRotation
)
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

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
