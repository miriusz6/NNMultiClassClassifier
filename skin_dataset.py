import os
import torch
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from helpers import get_files_in_dir
from torchvision.io import read_image


class SkinLesionData(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.labels_dir = label_dir
        self.count = len(get_files_in_dir(img_dir))

        self.transform_img_mask = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=45),
            v2.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9, 1.1), shear=10),
        ])
        self.transform_img = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        #self.transform_img_mask = v2.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(idx)+'.jpg')
        mask_path = os.path.join(self.labels_dir, str(idx)+'.png')
        image = read_image(img_path)
        mask = read_image(mask_path)
        #if self.transform_img_mask:
        # make mask have 3 channels instead of 1
        mask = mask.expand(3, -1, -1)
        imgNmsk = torch.stack([image, mask], dim=0)
        imgNmsk = self.transform_img_mask(imgNmsk)
        mask, image = imgNmsk[0], imgNmsk[1]
        image = self.transform_img(image)
        return image, mask
    

train_Dataset = SkinLesionData('data/train/images', 'data/train/masks')
train_loader = DataLoader(train_Dataset, batch_size=4, shuffle=True)

for i, (img, mask) in enumerate(train_loader):
    print(img.shape, mask.shape)
    plt.imshow(img[0].permute(1,2,0))
    plt.show()
    plt.imshow(mask[0].permute(1,2,0))
    plt.show()
    break