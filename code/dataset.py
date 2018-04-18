import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, roots, transform=None, loader=default_loader):
        images = []
        r_ids = []
        for r_id, root in enumerate(roots):
            for filename in os.listdir(root):
                if is_image_file(filename):
                    images.append('{}'.format(filename))
                    r_ids.append(r_id)
        self.roots = roots
        self.r_ids = r_ids
        self.imgs = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        root = self.roots[self.r_ids[index]]
        try:
            img = self.loader(os.path.join(root, filename))
        except:
            return torch.zeros((3, 32, 32))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
