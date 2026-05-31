import os
import random

import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from config import IMAGE_SIZE, BATCH_SIZE


# Выбранные классы

# Перенумерация классов



# =========================
# TRAIN AUGMENTATION
# =========================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.3403, 0.3121, 0.3214],
        std=[0.2724, 0.2608, 0.2669]
    )
])


# =========================
# TEST TRANSFORM
# =========================

test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.3403, 0.3121, 0.3214],
        std=[0.2724, 0.2608, 0.2669]
    )
])


class GTSRBDataset(Dataset):

    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None
    ):

        self.data = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        # Путь к изображению
        img_path = os.path.join(
            self.root_dir,
            row["Path"]
        )

        # Читаем изображение
        image = Image.open(img_path).convert("RGB")

        # =========================
        # ROI CROP
        # =========================

        x1 = int(row["Roi.X1"])
        y1 = int(row["Roi.Y1"])

        x2 = int(row["Roi.X2"])
        y2 = int(row["Roi.Y2"])

        # Crop по ROI
        image = image.crop((x1, y1, x2, y2))

        # =========================
        # LABEL
        # =========================

        original_label = int(row["ClassId"])

        label = int(row["ClassId"])

        # =========================
        # TRANSFORMS
        # =========================

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    train_csv,
    train_root,
    test_csv,
    test_root
):

    train_dataset = GTSRBDataset(
        csv_file=train_csv,
        root_dir=train_root,
        transform=train_transform
    )

    test_dataset = GTSRBDataset(
        csv_file=test_csv,
        root_dir=test_root,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader