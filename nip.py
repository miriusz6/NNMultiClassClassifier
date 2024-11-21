import torch
from torch import nn
from math import prod

# # #OrderedDict 
from collections import OrderedDict
from einops import rearrange, repeat



class PatchEmbedder(nn.Module):
    r"""
    Layer to take in the input image and do the following:
        1.  Transform grid of image into a sequence of patches.
            Number of patches are decided based on image height,width and
            patch height, width.
        2. Add cls token to the above created sequence of patches in the
            first position
        3. Add positional embedding to the above sequence(after adding cls)
        4. Dropout if needed
    """
    def __init__(self, config):
        super().__init__()
        # Example configuration
        #   Image c,h,w : 3, 224, 224
        #   Patch h,w : 16, 16
        image_height = config['image_height']
        image_width = config['image_width']
        im_channels = config['im_channels']
        emb_dim = config['emb_dim']
        patch_embd_drop = config['patch_emb_drop']
        
        self.patch_height = config['patch_height']
        self.patch_width = config['patch_width']

        self.mini_patch_factor = 2
        self.mini_patch_height = self.patch_height//2
        self.mini_patch_width = self.patch_width//2
        
        # Compute number of patches for positional parameters initialization
        #   num_patches = num_patches_h * num_patches_w
        #   num_patches = 224/16 * 224/16
        #   num_patches = 196
        num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)
        
        # This is the input dimension of the patch_embed layer
        # After patchifying the 224, 224, 3 image will be
        # num_patches x patch_h x patch_w x 3
        # Which will be 196 x 16 x 16 x 3
        # Hence patch dimension = 16 * 16 * 3
        patch_dim = im_channels * self.patch_height * self.patch_width

        patch_dim_single_ch = self.patch_height * self.patch_width
        self.channel_patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim_single_ch),
            nn.Linear(patch_dim_single_ch, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.patch_embed = nn.Sequential(
            # This pre and post layer norm speeds up convergence
            # Comment them if you want pure vit implementation
            nn.Linear(im_channels*emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        
        
        # Positional information needs to be added to cls as well so 1+num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(emb_dim))
        self.patch_emb_dropout = nn.Dropout(patch_embd_drop)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # This is doing the B, 3, 224, 224 -> (B, num_patches, patch_dim) transformation
        # B, 3, 224, 224 -> B, 3, 14*16, 14*16
        # B, 3, 14*16, 14*16 -> B, 3, 14, 16, 14, 16
        # B, 3, 14, 16, 14, 16 -> B, 14, 14, 16, 16, 3
        #  B, 14*14, 16*16*3 - > B, num_patches, patch_dim
        # out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
        #               ph=self.patch_height,
        #               pw=self.patch_width)
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) c (ph pw)',
                ph=self.patch_height,
                pw=self.patch_width,
                )
        
        out = self.channel_patch_embed(out)

        # sum over the channels
        #out = torch.sum(out, dim=2)
        out = out.flatten(2)
        out = self.patch_embed(out)


        # # # minipatches 3 channels 
        # # out = rearrange(out, 'b Ps (f sph) (g spw) c -> b Ps (f g) sph spw c',
        # #         sph=self.mini_patch_height,
        # #         spw=self.mini_patch_width,
        # #         )

        # # Add cls
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=batch_size)
        out = torch.cat((cls_tokens, out), dim=1)
        
        # # Add position embedding and do dropout
        out += self.pos_embed
        out = self.patch_emb_dropout(out)
        
        return out






class Nip(nn.Module):
    def __init__(self, 
                 device = 'cuda',
                 id_ = 0,
                 img_size = 128,
                 channels = 3,
                 patch_size = 32,
                 classes = 5,
                 embed_dim = 100
                 ):
        super().__init__()

        self.device = device
        self.id_ = id_
        self.img_size = img_size
        self.channels = channels
        self.patch_size = patch_size
        self.classes = classes
        self.embed_dim = embed_dim

        self.patches_pr_img = ((img_size//patch_size)**2)+1
        
        
        embed_config = {
            'image_height': img_size,
            'image_width': img_size,
            'im_channels': channels,
            'emb_dim': embed_dim,
            'patch_emb_drop': 0.1,
            'patch_height': patch_size,
            'patch_width': patch_size,
        }

        self.batch_norm = nn.BatchNorm2d(channels)

        self.patch_embed = PatchEmbedder(embed_config)

        self.body = nn.Sequential(
            nn.Linear(100,80),
            nn.LayerNorm(80),
            nn.Linear(80,65),
            nn.ReLU(),
            nn.Linear(65,10),
            nn.ReLU(),
        )

        self.body2 = nn.Sequential(
            nn.Linear(100,90),
            nn.LayerNorm(90),
            nn.Linear(90,80),
            nn.ReLU(),
            nn.Linear(80,70),
            nn.ReLU(),
            nn.Linear(70,60),
            nn.ReLU(),
            nn.Linear(60,10),
            nn.ReLU(),
        )

        self.body3 = nn.Sequential(
            nn.Linear(100,90),
            nn.LayerNorm(90),
            nn.Linear(90,80),
            nn.ReLU(),
            nn.Linear(80,70),
            nn.ReLU(),
            nn.Linear(70,60),
            nn.ReLU(),
            nn.Linear(60,50),
            nn.ReLU(),
            nn.Linear(50,10),
            nn.ReLU(),
        )


        self.head = nn.Sequential(
            #nn.LayerNorm(patches_pr_img*10),
            nn.Linear(self.patches_pr_img*10,(self.patches_pr_img//2)*10),
            nn.ReLU(),
            nn.Linear((self.patches_pr_img//2)*10,75),
            nn.ReLU(),
            nn.Linear(75,classes),
        )

    


        self.add_module('batch_norm', self.batch_norm)
        self.add_module('patch_embed', self.patch_embed)
        self.add_module('body', self.body)
        self.add_module('head', self.head)



    def forward(self, x):
        x = self.batch_norm(x)
        x = self.patch_embed(x)
        x1 = self.body(x)
        x2 = self.body2(x)
        x3 = self.body3(x)

        # stack x1, x2, x3 from BxPx3x50 to BxPx150
        x = torch.stack((x1,x2,x3), dim=2)
        #x = x.flatten(1)


        
        x = x.mean(dim=2)
        x = x.flatten(1)
        x = self.head(x)
        return x

    def print_sizes(self):
        pe = 0
        for _, param in self.patch_embed.named_parameters():
            pe+= param.numel()
        
        print("Patch embedder params: ",pe)

        b = 0
        for _, param in self.body.named_parameters():
            b+= param.numel()
        
        print("Body params: ",b)

        h = 0
        for _, param in self.head.named_parameters():
            h+= param.numel()
        
        print("Head params: ",h)

        print("Total params: ",pe+b+h)



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



model = Nip()
#model.print_sizes()
# # s = 0
# # for n,param in model.named_parameters():
# #     s+= param.numel()
# #     #print(n)

# # print("params: ",s)
# # # dummy img batch: 32x3x124x124
# # from figures_dataset import FiguresData

# # import torch

# # data_set = FiguresData(128, 2,augment = False)
# # data_loader = torch.utils.data.DataLoader(data_set, batch_size=2, shuffle=True)

# # model.print_sizes()

dummy_img = torch.rand(32,3,128,128)

r = model(dummy_img)


#print(r.shape)



# dummy_img = torch.rand(32,16,3,1024)

# dummy_lin = nn.Linear(3*1024, 50)



# r = dummy_lin(dummy_img)

# print(r.shape)

# dummy_img = torch.rand(32,3,16,16)
# dummy_conv = nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)

# r = dummy_conv(dummy_img)

# print(r.shape)

# #B x Ps x 4 x 16 x 16 x 3
# r = r.detach().cpu().numpy()

# img = r[0] 

# # 16 patches
# patches = [img[i] for i in range(0, img.shape[0])]
# print("Patches shape: ",img.shape)
# import numpy as np

# # 16 x 4 mini patches
# mini_patches = [np.array([[patches[i][0],patches[i][1]],[patches[i][2],patches[i][3]]]) for i in range(0, len(patches))]
# #print(print(patches[0][0].shape))
# print(mini_patches[0].shape)

# # visualize 
# import matplotlib.pyplot as plt


# # img: 16 x 4 x 16 x 16 x 3 to 64 x 16 x 16 x 3
# #img = rearrange(img, 'Ps Mps sph spw c -> (Mps Ps) sph spw c')
# #img = img.transpose(0,3,1,2)
# print(img.shape)




# a = range(64)
# k = 0
# for p in range(16):
#     fig, axs = plt.subplots(2,2, figsize=(10,10))
#     for i in range(2):
#         for j in range(2):
#             axs[i,j].imshow(img[p][i+j])
#             axs[i,j].axis('off')
#             k+=1
#     plt.show()