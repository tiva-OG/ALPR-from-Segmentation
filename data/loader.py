import cv2
import torch
import random
import logging
import numpy as np
import albumentations as A

from os import path
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset


def split_dataset(path, split=0.15):
    """ Splits the data into train and validation sets """
    
    img_ids = load_files(path)
    random.shuffle(img_ids)
    split = int(split * len(img_ids))
    train_ids, val_ids = img_ids[split:], img_ids[:split]

    return train_ids, val_ids

def load_files(path):
    """ Loads the ids(names) to the images present in `path` """
    exts = ['.jpg', '.jpeg', '.png']
    
    path = Path(path)
    # img_dir = path.joinpath('imgs')
    img_ids = [file.name for file in Path.iterdir(path) if file.suffix in exts]

    return img_ids

class LicensePlates(Dataset):
    """

    args:
    - path (str): root directory to the dataset containing `imgs` and `masks` folders
    - indices (iterable): file ids loaded from `load_files_id` or `split_dataset`; ids must match for both images and masks
    - mode (str): mode is a choice from [`train`, `test]
    - size (tuple, optional): size to reshape the data
    """

    def __init__(self, path, ids, mode, size=(572, 572)):
        super(LicensePlates, self).__init__()
        
        self.mode = mode
        self.size = size
        self.ids = ids
        self.path = Path(path)
        
        self.images = [self.path.joinpath('imgs', _id) for _id in ids]
        self.masks = [self.path.joinpath('masks', _id) for _id in ids]
        # self.imgs_dir = self.path.joinpath("./imgs")
        # self.masks_dir = self.path.joinpath("./masks")
        self.pixel_aug = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(0.005, 0.01), 
                contrast_limit=0.01, p=0.3
                ), 
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, p=0.3
                )
            ])
        self.transformation = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):
        img_path = str(self.images[i])
        mask_path = str(self.masks[i])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 0, 1, 0).astype("float32")        
        mask = cv2.resize(mask, self.size, cv2.INTER_NEAREST)

        if self.mode == "train":
            image = self.pixel_aug(image=image)["image"]

        image = self.transformation(image)
        mask = torch.from_numpy(mask)

        return {"image": image, "mask": mask}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    path = Path("dataset")
    train_ids, val_ids = split_dataset(path)
    dataset = LicensePlates(path, val_ids, "val")
    img, mask = dataset[9].values()
    
    logging.info(f"Shape of image: {img.shape}")
    logging.info(f"Shape of mask: {mask.shape}")
