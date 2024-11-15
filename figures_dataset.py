
from helpers import *
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch

class FiguresData(Dataset):
    def __init__(self, img_size, num_images,augment = False):
        self.img_size = img_size
        self.num_images = num_images
        # random rotation, nooise, flip, etc
        self.transform = v2.Compose([
            v2.GaussianBlur(3, (0.1, 1)),
            #v2.GaussianNoise(0.2),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(90),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])
        self.augment = augment
        
        self.labels = torch.zeros(num_images, dtype=torch.int64)
        self.images = torch.zeros(num_images, 3, img_size, img_size)

        for i in range(num_images//2):
            img = draw_random_triangle(self.img_size, (2,6))
            img = img/255
            img = img.astype(np.float32)
            img = torch.tensor(img).permute(2,0,1)
            lbl = torch.tensor(0)
            self.labels[i] = lbl
            self.images[i] = img
        
        for i in range(num_images//2,num_images):
            img = draw_random_circle(self.img_size, (6,20), (2,5))
            img = img/255
            img = img.astype(np.float32)
            img = torch.tensor(img).permute(2,0,1)
            lbl = torch.tensor(1)
            self.labels[i] = lbl
            self.images[i] = img
        
        # if self.augment:
        #     self.images = self.transform(self.images)


    def __len__(self):
        return self.num_images	

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_images:
            raise IndexError
        if self.augment:
            return self.transform(self.images[idx]), self.labels[idx]
        return self.images[idx], self.labels[idx]
        # if idx < 0 or idx >= self.num_images:
        #     raise IndexError
        # lbl, img = draw_random_figure(self.img_size, (1,5))
        # img = img/255
        # img = img.astype(np.float32)
        # img = torch.tensor(img).permute(2,0,1)
        # if self.augment:
        #     img = self.transform(img)

        # lbl = torch.tensor(lbl)
        # return img, lbl
    

# # from torch.utils.data import Dataset, DataLoader
# # import matplotlib.pyplot as plt
# # train_Dataset = FiguresData(100, 100)
# # train_loader = DataLoader(train_Dataset, batch_size=4, shuffle=True)

# # for i, (img, mask) in enumerate(train_loader):
# #     print(img.shape)
# #     print(mask.shape)
# #     plt.imshow(img[0].permute(1,2,0))
# #     plt.show()
# #     break