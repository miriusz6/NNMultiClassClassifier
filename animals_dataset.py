
from helpers import *
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch

from torchvision.io import read_image
#from tqdm.notebook import tqdm
from tqdm import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class AnimalsDataset(Dataset):
    def __init__(self, img_size, path ="animals_data", augment = False):
        self.img_size = img_size
        
        # random rotation, nooise, flip, etc
        self.transform = v2.Compose([
            # affine
            v2.RandomAffine(0, translate=(0.1, 0.1)),
            v2.RandomAffine(0, scale=(0.8, 1.2)),
            v2.RandomAffine(0, shear=(-10, 10)),
            # color
            #v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            #v2.GaussianNoise(0.1),
            v2.GaussianBlur(3, (0.1, 1)),
            # flip and rotation
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation((0,360)),
        ])

        
        cat_imgs = get_files_in_dir(path+"/cats")
        dog_imgs = get_files_in_dir(path+"/dogs")
        snks_imgs = get_files_in_dir(path+"/snakes")
        num_images = len(cat_imgs) + len(dog_imgs) + len(snks_imgs)
        self.num_images = num_images

        self.augment = augment
        self.labels = torch.zeros(num_images, dtype=torch.int64)
        self.images = torch.zeros(num_images, 3, img_size, img_size)

        animals_imgs = cat_imgs+dog_imgs+snks_imgs

        img_pbar = tqdm(range(len(animals_imgs)))

        for i in img_pbar:
            name = animals_imgs[i]
            if i < len(cat_imgs):
                animal = "cats"
                lbl = 0
            elif i < len(cat_imgs) + len(dog_imgs):
                animal = "dogs"
                lbl = 1
            else:
                animal = "snakes"
                lbl = 2
            img = read_image(f"{path}/{animal}/{name}")
            if i == 0 and img.shape[1] != img_size:
                self.images = torch.zeros(num_images, 3, img.shape[1], img.shape[2])
            img = img/255
            img = img.float()
            #lbl = torch.tensor(lbl)
            self.labels[i] = lbl
            self.images[i] = img

        if self.images[0].shape[0] != img_size:
            # use torch.resize
            self.images = torch.nn.functional.interpolate(self.images, size=(img_size, img_size), mode='bilinear', align_corners=False)
    

        self.labels = self.labels.to(device)
        self.images = self.images.to(device)
        self.images_no_augment = self.images.clone()


        if self.augment:
            self.new_augment()

    def new_augment(self):
        # iterate
        for i in range(100,self.num_images,100):
            self.images[i-100:i] = self.transform(self.images_no_augment[i-100:i])
            

    def __len__(self):
        return len(self.images)	

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_images:
            raise IndexError
        return self.images[idx], self.labels[idx]

    

# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# train_Dataset = AnimalsDataset(128)
# train_loader = DataLoader(train_Dataset, batch_size=4, shuffle=True)

# for i, (img, mask) in enumerate(train_loader):
#     print(img.shape)
#     print(mask.shape)
    
#     plt.imshow(img[0].detach().cpu().permute(1,2,0))
#     plt.show()
#    break