import torch
from torch import nn
from math import prod, sqrt

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

        self.cross_corr_cnt = 2
        
        
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
        # for 16x16 and 100 is 77k
        # patch_size x patch_size x channels x embed_dim
        self.patch_embed = PatchEmbedder(embed_config)

        correl_size = embed_dim # (self.embed_dim//self.patches_pr_img) *2

        # 20k
        self.correlation = nn.Linear(self.embed_dim*2, correl_size, device=device)

        self.cross_correlation = nn.Linear(correl_size*2, correl_size, device=device)

        # 20k
        self.apply_correlation = nn.Linear(self.embed_dim*2, self.embed_dim, device=device)
        

    def cartesian_prod_last_dim(self, x):
        batch = x.shape[0]
        patches = x.shape[1]
        embed = x.shape[2]
        
        # B x P x E -> B x P x 1 x E
        y = x.unsqueeze(2)
        # B x P x 1 x E -> B x P x P x E
        y = y.expand(batch,patches,patches,embed)
        # B x P x P x E -> B x (P x P) x E
        y  = y.reshape(batch,patches*patches,embed)

        # B x (P x P) x E -> B x (P x P) x E
        # but the last dim is no longer the same 
        # (one repeated) embedding
        z = rearrange(y, 'b (ps pss) e -> b (pss ps) e',
                            ps = patches, pss = patches)
        
        # cart_prod
        # B x (P x P) x E -> B x P x P x 2E
        ret = torch.cat((y,z),dim=2)
        ret = ret.reshape(batch,patches,patches,2*embed)
        return ret
        
    def apply_layer_piecewise(self, x, layer, piece_cnt = 2):
        i = int(sqrt(x.shape[1]))
        pc = piece_cnt
        for j in range(i):
            # B x P x P x 2E -> B x P x P/2 x 2 x 2E
            x = x.reshape(x.shape[0],x.shape[1],x.shape[2]//pc,2,x.shape[3])
            # B x P x P/2 x 4E
            x = x.flatten(3)
            # B x P x P/2 x 4E -> B x P x P/2 x 2E
            x = layer(x)
        
        x = x.squeeze()
        return x

    def forward(self, x):
        B = x.shape[0]
        x = self.batch_norm(x)
        # B x P x E
        embeds = self.patch_embed(x)

        # B x P x P x 2E 
        # also dosnt consider cls tokens
        embeds_cart = self.cartesian_prod_last_dim(
            embeds[:,0:self.patches_pr_img-1,:])
        # B x P x P x E
        correl = self.correlation(embeds_cart)
        # B x P x P x E -> B x P x E
        cross_correl = self.apply_layer_piecewise(correl, self.cross_correlation, self.cross_corr_cnt)
        # 1 x E
        cls = embeds[:,embeds.shape[1]-1,:]
        # B x E
        cls = cls.expand(B,self.embed_dim)
        # B x 1 x E
        cls = cls.unsqueeze(1)
        # B x P x E
        # where P is with cls tokens
        cross_correl = torch.cat((cross_correl,cls),dim=1)





        return cross_correl

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


dummy_img = torch.rand(32,3,128,128, device=device)

r = model(dummy_img)

print(r.shape)
