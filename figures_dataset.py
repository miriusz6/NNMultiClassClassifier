import torch
from helpers import *
from torch.utils.data import Dataset
from torchvision.transforms import v2

class FiguresData(Dataset):
    def __init__(self, img_size, num_images):
        self.img_size = img_size
        self.num_images = num_images

        # random rotation, nooise, flip, etc
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(90),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.RandomAffine(0, translate=(0.1, 0.1)),
            v2.RandomAffine(0, scale=(0.9, 1.1)),
            v2.RandomAffine(0, shear=10),
            v2.ToTensor()
        ])

        self.images = []
        self.labels = []
        for i in range(num_images//5):
            img = draw_random_circle(img_size,(5, 30), (1, 5))
            self.images.append(img)
            self.labels.append(0)
            img = draw_random_rectangle(img_size, (1, 5))
            self.images.append(img)
            self.labels.append(1)
            # img = draw_random_elipse(img_size, (1, 5))
            # self.images.append(img)
            # self.labels.append(2)
            # img = draw_random_line(img_size, (1, 5))
            # self.images.append(img)
            # self.labels.append(3)
            # img = draw_random_triangle(img_size, (1, 5))
            # self.images.append(img)
            # self.labels.append(4)

        self.images = np.array(self.images)
        self.images = self.images/255
        self.images = self.images.astype(np.float32)
        
        self.images = torch.tensor(self.images).permute(0,3,1,2)
        self.images = self.transform(self.images)

        self.labels = torch.tensor(self.labels)
        self.count = len(self.images)

        

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    

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