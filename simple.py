import torch
from torch import nn
from math import prod

# # #OrderedDict 
from collections import OrderedDict
from einops import rearrange, repeat



class PatchEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Example configuration
        #   Image c,h,w : 3, 224, 224
        #   Patch h,w : 16, 16
        image_height = config['image_height']
        image_width = config['image_width']
        im_channels = config['im_channels']
        emb_dim = config['emb_dim']
        
        self.patch_height = config['patch_height']
        self.patch_width = config['patch_width']

        device = 'cuda'
        
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
        
        self.patch_embed = nn.Linear(patch_dim, emb_dim, device=device)
        self.patch_embed.weight.data.fill_(1.0)
        self.patch_embed.bias.data.fill_(1.0)
        
        # Positional information needs to be added to cls as well so 1+num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(emb_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
  
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                      ph=self.patch_height,
                      pw=self.patch_width)
        out = self.patch_embed(out)
        
        # Add cls
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=batch_size)
        cls_tokens = cls_tokens.to(x.device)
        out = torch.cat((cls_tokens, out), dim=1)
        
        # Add position embedding and do dropout
        out += self.pos_embed.to(x.device)
        return out

    


# class SimpleTrainer(nn.Module):
#     def __init__(self, device = 'cuda'):
#         super().__init__()
#         self.layer = layer

#         self.cols = layer[1]
#         self.rows = layer_dims[0]

#         self.l = nn.Sequential(
#             nn.Linear(self.rows, self.rows*2, device=device),
#             nn.ReLU(),
#             nn.Linear(self.rows*2, self.rows*4, device=device),
#             nn.ReLU(),
#             nn.Linear(self.rows*4, self.rows*2, device=device),
#             nn.ReLU(),
#             nn.Linear(self.rows*2, self.rows, device=device),
#         )

        

#         self.device = device

#     def forward(self, x):
#         out = torch.zeros(self.rows, self.cols, device=self.device)
#         for col in range(self.cols):
#             out[:,col] = self.l(x[:,col])
#         return out
        


class SimpleModel(nn.Module):
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

        self.batch_norm = nn.BatchNorm2d(channels, device=device)

        self.patch_embed = PatchEmbedder(embed_config)

        self.lin1 = nn.Linear(embed_dim, 100, device=device)
        self.lin2 = nn.Linear(100, 60, device=device)
        self.lin3 = nn.Linear(60, 10, device=device)
        self.combine = nn.Linear(self.patches_pr_img*10, self.patches_pr_img*5, device=device)
        self.head = nn.Linear(self.patches_pr_img*5, classes, device=device)

        
        self.grad_false()

        ps = self.patch_embed.patch_embed.weight.data
        self.cols = ps.shape[1]
        self.rows = ps.shape[0]
        self.nope = nn.Sequential(
            nn.Linear(self.rows, self.rows*2, device=device),
            nn.ReLU(),
            nn.Linear(self.rows*2, self.rows*4, device=device),
            nn.ReLU(),
            nn.Linear(self.rows*4, self.rows*2, device=device),
            nn.ReLU(),
            nn.Linear(self.rows*2, self.rows, device=device),
        )

        self.set_to_ones()



    def forward(self, x):
        #x = self.batch_norm(x)
        ps = self.patch_embed.patch_embed.weight.data
        # for col in range(self.cols):
        #     a = ps[:,col]
        #     b = self.nope(a)
        #     ps[:,col] = self.nope(b)

        a = ps.permute(1,0)
        b = self.nope(a)
        #self.patch_embed.patch_embed.weight.data = b.permute(1,0)
        x = self.patch_embed(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.combine(x.flatten(1))
        x = self.head(x)

        return x

    def set_to_ones(self):
        self.lin1.weight.data.fill_(1.0)
        self.lin1.bias.data.fill_(1.0)
        self.lin2.weight.data.fill_(1.0)
        self.lin2.bias.data.fill_(1.0)
        self.lin3.weight.data.fill_(1.0)
        self.lin3.bias.data.fill_(1.0)
        self.combine.weight.data.fill_(1.0)
        self.combine.bias.data.fill_(1.0)
        self.head.weight.data.fill_(1.0)
        self.head.bias.data.fill_(1.0)

        # set nope to ones
        for l in self.nope:
            if isinstance(l, nn.Linear):
                l.weight.data.fill_(1.0)
                l.bias.data.fill_(1.0)
    
    def grad_false(self):
        self.patch_embed.patch_embed.requires_grad = False
        self.lin1.requires_grad = False
        self.lin2.requires_grad = False
        self.lin3.requires_grad = False
        self.combine.requires_grad = False
        self.head.requires_grad = False

    def print_sizes(self):
        print("Patch Embed: ",self.patch_embed.patch_embed.weight.shape, " Grad: ", self.patch_embed.patch_embed.weight.requires_grad)
        print("Lin1: ",self.lin1.weight.shape, " Grad: ", self.lin1.weight.requires_grad)
        print("Lin2: ",self.lin2.weight.shape, " Grad: ", self.lin2.weight.requires_grad)
        print("Lin3: ",self.lin3.weight.shape, " Grad: ", self.lin3.weight.requires_grad)
        print("Combine: ",self.combine.weight.shape, " Grad: ", self.combine.weight.requires_grad)
        print("Head: ",self.head.weight.shape, " Grad: ", self.head.weight.requires_grad)




device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



model = SimpleModel()
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

dummy_img = torch.rand(32,3,128,128, device=device)

r = model(dummy_img)
# model.set_to_ones()
# model.grad_false()

print(r.shape)

# s_t = SimpleTrainer([128,5])

# dummy_ten = torch.rand(1,128,5)

# r = s_t(dummy_ten)

# print(r.shape)