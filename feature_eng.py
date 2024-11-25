import torch
from torch import nn
from math import prod, sqrt, log

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
        #patch_dim = im_channels * self.patch_height * self.patch_width
        patch_dim = self.patch_height * self.patch_width
        
        self.patch_embed = nn.Linear(patch_dim, emb_dim, device=device)
        self.patch_embed.weight.data.fill_(1.0)
        self.patch_embed.bias.data.fill_(1.0)
        
        # Positional information needs to be added to cls as well so 1+num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        #self.cls_token = nn.Parameter(torch.randn(emb_dim))

        self.layernorm = nn.LayerNorm(emb_dim, device=device)
        
    def forward(self, x):
  
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b c (nh nw) (ph pw)',
                      ph=self.patch_height,
                      pw=self.patch_width)
        
        out = out.sum(dim=1)

        out = self.patch_embed(out)
        
        # Add cls
        #cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=batch_size)
        #cls_tokens = cls_tokens.to(x.device)
        #out = torch.cat((cls_tokens, out), dim=1)
        
        # Add position embedding and do dropout
        out += self.pos_embed.to(x.device)
        # layer norm
        out = self.layernorm(out)
        return out




class CrossCorrelator(nn.Module):
    def __init__(self, 
                 device = 'cuda',
                 embed_dim = 100
                 ):
        super().__init__()

        self.device = device
        self.embed_dim = embed_dim

        self.cross_corr_cnt = 2
        
        correl_size = embed_dim

        self.cross_correlation = nn.Sequential(
            nn.Linear(correl_size*2, int(correl_size*1.5), device=device),
            nn.ReLU(),
            nn.Linear(int(correl_size*1.5),correl_size, device=device),
            nn.ReLU(),#nn.LayerNorm(correl_size, device=device),
        )

        self.cross_path = nn.Sequential(
            nn.Linear(self.embed_dim*2, int(embed_dim*1.5), device=device),
            nn.ReLU(),
            nn.Linear(int(embed_dim*1.5), self.embed_dim, device=device),
            #nn.ReLU(),#nn.LayerNorm(self.embed_dim, device=device),
        )

        self.correl_weight = nn.Parameter(torch.tensor(0.5))
        self.cross_correl_weight = nn.Parameter(torch.tensor(0.5))

        # add modules to the model
        self.add_module('cross_correlation', self.cross_correlation)
        self.add_module('cross_path', self.cross_path)

    
        

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
        z = z * self.correl_weight
        # cart_prod
        # B x (P x P) x E -> B x P x P x 2E
        ret = torch.cat((y,z),dim=2)
        ret = ret.reshape(batch,patches,patches,2,embed)
        return ret
        
    def apply_layer_piecewise(self, x, layer, piece_cnt = 2, dim = 2, early_dim_stop = -1):
        i = log(x.shape[dim], piece_cnt)
        
        if i != int(i):
            raise Exception("Not implemented for this case yet")
        
        if early_dim_stop != -1 :
            i -= log(early_dim_stop, piece_cnt)
        i = int(i)
        pc = piece_cnt
        for j in range(i):
            # B x P x P x 2E -> B x P x P/2 x 2 x 2E
            d_ = max(1,x.shape[dim]//pc)
            dims = x.shape[0:dim] + (d_, pc) + x.shape[dim+1:]
                
            #x = x.reshape(x.shape[0],x.shape[1],x.shape[2]//pc,2,x.shape[3])
            x = x.reshape(dims)
            # B x P x P/2 x 4E
            x = x.flatten(dim+1)
            # B x P x P/2 x 4E -> B x P x P/2 x 2E
            x = layer(x)
        
        x = x.squeeze()
        return x
    
    def forward(self, x):
        # B x P x E
        short = x
        # B x P x P x 2E 
        embeds_cart = self.cartesian_prod_last_dim(x)
        # B x P x P x E
        correl = embeds_cart.sum(dim=3)
        # B x P x P x E -> B x P x E
        cross_correl = self.apply_layer_piecewise(correl, self.cross_correlation, self.cross_corr_cnt, dim = 2)

        correl_aware = short + (cross_correl*self.cross_correl_weight)
        cross_path = self.apply_layer_piecewise(correl_aware, self.cross_path, 2, dim = 1,
                                                early_dim_stop=4)
        cross_path = cross_path.flatten(1)
        
        return cross_path




class FModel(nn.Module):
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

        self.cross_path = nn.Sequential(
            nn.Linear(self.embed_dim*2, int(embed_dim*1.5), device=device),
            nn.ReLU(),
            nn.Linear(int(embed_dim*1.5), self.embed_dim, device=device),
            nn.LayerNorm(self.embed_dim, device=device),
            nn.ReLU(),
        )

        self.cross_path2 = nn.Sequential(
            nn.Linear(self.embed_dim*2, int(embed_dim*1.5), device=device),
            nn.ReLU(),
            nn.Linear(int(embed_dim*1.5), self.embed_dim, device=device),
            nn.LayerNorm(self.embed_dim, device=device),
            nn.ReLU(),
        )

        
        self.head = nn.Sequential( 
            nn.Linear(self.embed_dim*4, self.embed_dim*2, device=device),
            nn.ReLU(),
            nn.Linear(self.embed_dim*2,self.embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim//2, device=device),
            nn.ReLU(),
            nn.Linear(self.embed_dim//2,self.classes, device=device),
        )

        self.correlator = nn.Sequential(
            CrossCorrelator(device=device, embed_dim=embed_dim),
            nn.LayerNorm(embed_dim*4, device=device),
            nn.ReLU(),
        ) 
        # self.correlator2 = nn.Sequential(
        #     CrossCorrelator(device=device, embed_dim=embed_dim),
        #     nn.LayerNorm(embed_dim*4, device=device),
        #     nn.ReLU(),
        # )

        self.comb_pred = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.classes*2, self.classes*2, device=device),
            nn.ReLU(),
            nn.Linear(self.classes*2,self.classes, device=device),
        )

        self.feat_correl = nn.Sequential(
            nn.LayerNorm(embed_dim*4, device=device),
            nn.Linear(embed_dim*4, embed_dim*2, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim, device=device),
            nn.LayerNorm(embed_dim, device=device),
            nn.ReLU(),
        )

        self.feat_downsize = nn.Sequential(
            nn.Linear(embed_dim, 4, device=device),
            nn.ReLU(),
        )
            

        self.seq1 = nn.Sequential(
            nn.Linear(self.embed_dim*8, self.embed_dim*4, device=device),
            nn.ReLU(),
            nn.Linear(self.embed_dim*4, self.embed_dim*4, device=device),
            nn.LayerNorm(self.embed_dim*4, device=device),
            nn.ReLU(),
        )

        self.cor1_w = nn.Parameter(torch.tensor(0.5))
        self.cor2_w = nn.Parameter(torch.tensor(0.5))
        self.main_w = nn.Parameter(torch.tensor(0.5))
        self.f_corr_w = nn.Parameter(torch.tensor(0.5))

        self.add_module('batch_norm', self.batch_norm)
        self.add_module('patch_embed', self.patch_embed)
        self.add_module('head', self.head)
        self.add_module('correlator', self.correlator)
        #self.add_module('correlator2', self.correlator2)
        self.add_module('cross_path', self.cross_path)


        self.mode = 0

    def cartesian_prod_last_dim_features(self, x):
        batch = x.shape[0]
        patches = x.shape[1]
        embed = x.shape[2]
        
        # B x P x E -> B x P x 1 x E
        y = x.unsqueeze(2)
        # B x P x 1 x E -> B x P x P x E
        y = y.expand(batch,patches,embed, embed)
        # B x P x P x E -> B x (P x P) x E
        y  = y.reshape(batch,patches,embed * embed)

        # B x (P x P) x E -> B x (P x P) x E
        # but the last dim is no longer the same 
        # (one repeated) embedding
        z = rearrange(y, 'b ps (e ee) -> b ps (ee e)',
                            e = embed, ee = embed)
        #z = z * self.correl_weight
        # cart_prod
        # B x (P x P) x E -> B x P x P x 2E
        ret = torch.cat((y,z),dim=2)
        ret = ret.reshape(batch,patches,2,embed*embed)
        return ret

    def apply_layer_piecewise(self, x, layer, piece_cnt = 2, dim = 2, early_dim_stop = -1):
        i = log(x.shape[dim], piece_cnt)
        
        if i != int(i):
            raise Exception("Not implemented for this case yet")
        
        if early_dim_stop != -1 :
            i -= log(early_dim_stop, piece_cnt)
        i = int(i)
        pc = piece_cnt
        for j in range(i):
            # B x P x P x 2E -> B x P x P/2 x 2 x 2E
            d_ = max(1,x.shape[dim]//pc)
            dims = x.shape[0:dim] + (d_, pc) + x.shape[dim+1:]
                
            #x = x.reshape(x.shape[0],x.shape[1],x.shape[2]//pc,2,x.shape[3])
            x = x.reshape(dims)
            # B x P x P/2 x 4E
            x = x.flatten(dim+1)
            # B x P x P/2 x 4E -> B x P x P/2 x 2E
            x = layer(x)
        
        x = x.squeeze()
        return x
    
    def feature_correl(self, x):
        cart = self.cartesian_prod_last_dim(x)
        v_dim = (x.shape[0],x.shape[1]*x.shape[2]) + x.shape[3:]
        xA = cart.view(v_dim)[:,:,0,:]
        xB = cart.view(v_dim)[:,:,1,:]
        return (xA * xB)


    def switch_mode(self, mode):
        self.mode = mode
        if mode == 0:
            for param in self.parameters():
                param.requires_grad = True
            self.req_grad(self.seq1,False)
            self.req_grad(self.cross_path2,False)
        if mode == 1:
            for param in self.parameters():
                param.requires_grad = False
            self.req_grad(self.seq1,True)
            self.req_grad(self.seq1,True)
    
    def req_grad(self, l, req = True):
        for param in l.parameters():
            param.requires_grad = req
        

    def forward(self, x):

        x = self.batch_norm(x)
        # B x P x E
        embeds = self.patch_embed(x)

        cross_path = self.apply_layer_piecewise(embeds, self.cross_path, 2, dim = 1,
                                                early_dim_stop=4)
        cross_path = cross_path.flatten(1)

        
        
        #cross_path_corr = self.correlator(embeds)

        # y = self.cartesian_prod_last_dim_features(embeds)
        # y = y[:,:,0,:] * y[:,:,1,:]
        # y = self.feat_downsize(y.view(y.shape[0],
        #                               y.shape[1],self.embed_dim,self.embed_dim))
        # y = y.flatten(-2)
        # y = self.feat_correl(y)

        # cross_path_y = self.apply_layer_piecewise(y, self.cross_path, 2, dim = 1,
        #                                         early_dim_stop=4)
        # cross_path_y = cross_path_y.flatten(1)

        # cross_path_corr_y = self.correlator(y)

        if self.mode == 1:    
            cross_path2 = self.apply_layer_piecewise(embeds, self.cross_path2, 2, dim = 1,
                                                    early_dim_stop=4)
            cross_path2 = cross_path2.flatten(1)

            cp =  torch.cat((cross_path,cross_path2),dim=1)
            cp = self.seq1(cp)
            pred = self.head(cp)
        else:
            pred = self.head(cross_path)

        
        #pred2 = self.head(cross_path_y)
        #pred3 = self.head(cross_path_corr)
        #pred4 = self.head(cross_path_corr_y)
        #p = torch.cat((pred1*self.main_w,pred2*self.f_corr_w, pred3*self.cor1_w,pred4*self.cor2_w),dim=1)
        #p = torch.cat((pred1*self.main_w, pred3*self.cor1_w),dim=1)
        #pred = self.comb_pred(p) #(pred1*self.main_w) + (pred2 * self.cor1_w) + (pred3 * self.cor2_w)
        
        
        
        return pred


    def print_sizes(self):
        print("Patch Embed: ",self.patch_embed.patch_embed.weight.shape, " Grad: ", self.patch_embed.patch_embed.weight.requires_grad)
        print("Pos Embed: ",self.patch_embed.pos_embed.shape, " Grad: ", self.patch_embed.pos_embed.requires_grad)
        print("Head: ",self.head[0].weight.shape, " Grad: ", self.head[0].weight.requires_grad)
        c = 0
        for name, param in self.named_parameters():
            print(name, param.shape, param.numel())
            c += param.numel()
        print("Total - embedder: ", c - self.patch_embed.patch_embed.weight.numel() - self.patch_embed.pos_embed.numel())
        print("Total: ", c)
        


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



model_config = {
    "img_size": 128,
    "classes": 5,
    "id_" : 1,
    "patch_size" : 16,
    "embed_dim" : 100,
}

model = FModel(**model_config)


dummy_img = torch.rand(32,3,128,128, device=device)

r = model(dummy_img)
print(r.shape)

model.switch_mode(0)
model.switch_mode(1)
#print(model.patch_embed.requires_grad)
