import torch
from torch import nn
from embed import PatchEmbedding

# # #OrderedDict 
from collections import OrderedDict
# # a = OrderedDict({'0': nn.ReLU(), '1':nn.Linear(50,40, bias=True)})
# # nn.Sequential( a )



class patch_pipe(nn.Module):
        def __init__(self, 
                    sizes,
                    sizes_pr_level,
                    forks,
                    ):
            super().__init__()

            layers = OrderedDict()

            # leaf
            if len(forks) < 1:
                my_sizes = sizes
            # not leaf
            else:
                my_sizes = sizes[:sizes_pr_level]

            for i in range(len(my_sizes)-1):
                dim1 = my_sizes[i]
                dim2 = my_sizes[i+1]
                layers['Lin'+str(i)] = nn.Linear(dim1,dim2, bias=True)
                layers['ReLu'+str(i)] = nn.ReLU()
            
            layers['Norm'] = nn.LayerNorm(my_sizes[-1])
            
            self.layers = nn.Sequential(layers)
            self.children_pipes = []
            
            if len(forks) > 0:
                children_cnt = forks[0]
                for i in range(children_cnt):
                    kid = patch_pipe(sizes[sizes_pr_level-1:], sizes_pr_level, forks[1:])
                    self.children_pipes.append(kid)
                    self.add_module('child'+str(i), kid)


        def forward(self, x):
            x = self.layers(x)
            out = []
            for child in self.children_pipes:
                out = out + child(x)
            if len(out) == 0:
                return [x]
            return out
            



class patch_pipe_root(nn.Module):
    def __init__(self, 
                emed_dim,
                forks,
                depth,
                out_dim,
                 ):
        super().__init__()

        self.sizes_pr_level = depth // len(forks)
        size_step = -int((emed_dim-out_dim)/depth)
        sizes = [i for i in range(emed_dim,out_dim, size_step)]
        sizes = sizes[0:depth] + [out_dim]

        children_cnt = forks[0]
        self.children_pipes = []
        for i in range(children_cnt):
            kid = patch_pipe(
                sizes,
                self.sizes_pr_level,
                forks[1:])
            self.children_pipes.append(kid)
            self.add_module('child'+str(i), kid)

        
    def forward(self, x):
        out = []
        for child in self.children_pipes:
            out = out + child(x)
        return out

class tit(nn.Module):
    def __init__(self, 
                 img_size = 128,
                 channels = 3,
                 patch_size = 32,
                 classes = 5,
                 embed_dim = 50,
                 patch_pipe_root_forks = [2,2], # must be >1 forks patch_pipe
                 patch_pipe_root_depth = 10,
                 patch_pipe_root_out_dim = 15,
                 patch_pipe_tree_merges = [2,2], # must be patch_pipe_blocks % necks_input_pipes == 0,
                 ):
        super().__init__()

        self.img_size = img_size
        self.channels = channels
        self.patch_size = patch_size
        self.classes = classes
        self.embed_dim = embed_dim
        self.patch_pipe_forks = patch_pipe_root_forks
        self.patch_pipe_merges = patch_pipe_tree_merges
        
        
        embed_config = {
            'image_height': img_size,
            'image_width': img_size,
            'im_channels': channels,
            'emb_dim': embed_dim,
            'patch_emb_drop': 0.1,
            'patch_height': patch_size,
            'patch_width': patch_size,
        }

        self.patch_embed = PatchEmbedding(embed_config)

        self.root = patch_pipe_root(embed_dim, 
                                    patch_pipe_root_forks, 
                                    patch_pipe_root_depth, 
                                    patch_pipe_root_out_dim)
        
        self.add_module('patch_embed', self.patch_embed)
        self.add_module('root', self.root)

        # # # self.process_patch1 = nn.Sequential(
        # # #     nn.ReLU(),
        # # #     nn.Linear(50,40, bias=True),
        # # #     nn.ReLU(),
        # # #     nn.Linear(40,30, bias=True),
        # # #     nn.ReLU(),
        # # #     nn.Linear(30,25, bias=True),
        # # #     nn.ReLU(),
        # # #     nn.Linear(25,15, bias=True),
        # # #     nn.LayerNorm(15),
        # # # )

        # # # self.process_patch2 = nn.Sequential(
        # # #     nn.ReLU(),
        # # #     nn.Linear(50,40, bias=True),
        # # #     nn.ReLU(),
        # # #     nn.Linear(40,30, bias=True),
        # # #     nn.ReLU(),
        # # #     nn.Linear(30,25, bias=True),
        # # #     nn.ReLU(),
        # # #     nn.Linear(25,15, bias=True),
        # # #     nn.LayerNorm(15),
        # # # )

        # # # # self.process_patch3 = nn.Sequential(
        # # # #     nn.ReLU(),
        # # # #     nn.Linear(50,40, bias=True),
        # # # #     nn.ReLU(),
        # # # #     nn.Linear(40,30, bias=True),
        # # # #     nn.ReLU(),
        # # # #     nn.Linear(30,25, bias=True),
        # # # #     nn.ReLU(),
        # # # #     nn.Linear(25,15, bias=True),
        # # # #     nn.LayerNorm(15),
        # # # # )

        # # # # self.process_patch4 = nn.Sequential(
        # # # #     nn.ReLU(),
        # # # #     nn.Linear(50,40, bias=True),
        # # # #     nn.ReLU(),
        # # # #     nn.Linear(40,30, bias=True),
        # # # #     nn.ReLU(),
        # # # #     nn.Linear(30,25, bias=True),
        # # # #     nn.ReLU(),
        # # # #     nn.Linear(25,15, bias=True),
        # # # #     nn.LayerNorm(15),
        # # # # )

        # # # self.neck1 = nn.Sequential(
        # # #     nn.LayerNorm(15*17*2),
        # # #     nn.Linear(15*17*2,100, bias=True),
        # # #     nn.ReLU(),
        # # #     nn.Linear(100,50, bias=True),
        # # # )

        # # # # self.neck2 = nn.Sequential(
        # # # #     nn.LayerNorm(15*17*2),
        # # # #     nn.Linear(15*17*2,100, bias=True),
        # # # #     nn.ReLU(),
        # # # #     nn.Linear(100,50, bias=True),
        # # # # )


        # # # self.last = nn.Sequential(
        # # #     nn.LayerNorm(50),
        # # #     nn.Linear(50,50, bias=True),
        # # #     nn.ReLU(),
        # # #     nn.Linear(50,5, bias=True),
        # # # )

        



    def forward(self, x):
        x = self.patch_embed(x)
        x = self.root(x)
        return x

        # # # B, C, H, W -> B, P, E
        # # x = self.patch_embed(x)
        
        # # x1 = self.process_patch1(x).flatten(1)
        # # x2 = self.process_patch2(x).flatten(1)
        # # # x3 = self.process_patch3(x).flatten(1)
        # # # x4 = self.process_patch4(x).flatten(1)
        
        # # X1 = torch.cat([x1,x2],1)
        # # #X2 = torch.cat([x3,x4],1)

        # # X1 = self.neck1(X1)
        # # #X2 = self.neck2(X2)

        # # #X = torch.cat([X1,X2],1)
        # # X = X1
        # # X = self.last(X)

        # # return X

