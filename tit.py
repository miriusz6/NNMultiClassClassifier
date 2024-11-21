import torch
from torch import nn
from embed import PatchEmbedding
from math import prod

# # #OrderedDict 
from collections import OrderedDict

class RootLevel(nn.Module):
        def __init__(self, 
                    sizes,
                    sizes_pr_level,
                    forks,
                    ):
            super().__init__()
            # leaf
            if len(forks) < 1:
                my_sizes = sizes
            # not leaf
            else:
                my_sizes = sizes[:sizes_pr_level]

            self.pipe = Pipe(my_sizes)
            self.children_pipes = []
            
            if len(forks) > 0:
                children_cnt = forks[0]
                for i in range(children_cnt):
                    kid = RootLevel(sizes[sizes_pr_level-1:], sizes_pr_level, forks[1:])
                    self.children_pipes.append(kid)
                    self.add_module('child'+str(i), kid)


        def forward(self, x):
            x = self.pipe(x)
            out = []
            for child in self.children_pipes:
                out = out + child(x)
            if len(out) == 0:
                return [x]
            return out
            
class Root(nn.Module):
    def __init__(self, 
                emed_dim,
                forks,
                depth,
                out_dim,
                 ):
        super().__init__()

        self.sizes_pr_level = depth // len(forks)
        size_step = int((emed_dim-out_dim)/depth)
        size_step = max(1,size_step)
        sizes = [i for i in range(emed_dim,out_dim, -size_step)]
        sizes = sizes[0:depth] + [out_dim]

        short = depth - len(sizes) 
        if short > 0:
            sizes = sizes + [out_dim]*short

        children_cnt = forks[0]
        self.children_pipes = []
        for i in range(children_cnt):
            kid = RootLevel(
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


class Pipe(nn.Module):
        def __init__(self, 
                    sizes
                    ):
            super().__init__()

            layers = OrderedDict()
            for i in range(len(sizes)-1):
                dim1 = sizes[i]
                dim2 = sizes[i+1]
                layers['Lin'+str(i)] = nn.Linear(dim1,dim2, bias=True)
                layers['ReLu'+str(i)] = nn.ReLU()
            
            #layers['Norm'] = nn.LayerNorm(sizes[-1])
            self.layers = nn.Sequential(layers)

        def forward(self, x):
            x = self.layers(x)
            return x


class TreeLevel(nn.Module):
    def __init__(self,
            pipe_cnt,
            leafs_pr_pipe,
            sizes,
            level
        ):
        super().__init__()

        self.pipes = []
        self.pipe_cnt = pipe_cnt
        self.level = level
        self.leafs_pr_pipe = leafs_pr_pipe

        for i in range(pipe_cnt):
            pipe = Pipe(sizes)
            self.pipes.append(pipe)
            self.add_module('pipe'+str(i), pipe)

    def forward(self, x):
        out = []
        for i,pipe in enumerate(self.pipes):
            if len(self.pipes) == 1:
                pipe_input = torch.cat(x, 1)
            else:
                pipe_input = x[i*self.leafs_pr_pipe:(i+1)*self.leafs_pr_pipe]
                pipe_input = torch.cat(pipe_input, 1)
            out.append(pipe(pipe_input))
        return out


class Tree(nn.Module):
    def __init__(self, 
                emed_dim,
                merges,
                depth,
                out_dim,
                leafs_on_top,
                patch_cnt,
                 ):
        super().__init__()
        self.input_pipes = merges[0]
        self.sizes_pr_level = depth // len(merges)
        size_step = int((emed_dim-out_dim)/depth)
        size_step = max(1,size_step)
        sizes = [i for i in range(emed_dim,out_dim, -size_step)]
        sizes = sizes[0:depth] + [out_dim]

        short = depth - len(sizes) 
        if short > 0:
            sizes = sizes + [out_dim]*short
        

        self.levels = []
        for i,merg in enumerate(merges):
            leafs_on_lvl = leafs_on_top//(prod(merges[:i]))
            pipes_cnt_on_lvl = leafs_on_lvl//merg
            leafs_pr_pipe = leafs_on_lvl//pipes_cnt_on_lvl
            fst_s = (i*self.sizes_pr_level)-1
            lst_s = (i+1)*self.sizes_pr_level
            # first level
            if i == 0:
                fst_s = 0
            # last level
            if i == len(merges)-1:
                lst_s = len(sizes)
            sizes_on_lvl = sizes[fst_s:lst_s]
            sizes_on_lvl[0] = sizes_on_lvl[0]*leafs_pr_pipe
            if i == 0:
                sizes_on_lvl[0] = sizes_on_lvl[0]*patch_cnt
            
            level = TreeLevel(pipes_cnt_on_lvl,leafs_pr_pipe, sizes_on_lvl,i)
            self.levels.append(level)
            self.add_module('level'+str(i), level)
        
        
        
    def forward(self, x):
        # flatten
        x = [t.flatten(1) for t in x]
        for level in self.levels:
            x = level(x)
        return x[0]

class tit(nn.Module):
    def __init__(self, 
                 device = 'cuda',
                 id = 0,
                 img_size = 128,
                 channels = 3,
                 patch_size = 32,
                 classes = 5,
                 embed_dim = 50,
                 patch_pipe_root_forks = [2,2], # must be >1 forks patch_pipe
                 patch_pipe_root_depth = 10,
                 patch_pipe_root_out_dim = 25,
                 patch_pipe_tree_merges = [2,2], # must be patch_pipe_blocks % necks_input_pipes == 0,
                 patch_pipe_tree_depth = 10,
                 patch_pipe_tree_out_dim = 10,
                 ):
        super().__init__()

        if ( prod(patch_pipe_root_forks) != prod(patch_pipe_tree_merges)):
            raise ValueError("Product of forks must be equal product of merges! e.g. [2,4,2] and [4,4]")

        if (embed_dim < patch_pipe_root_out_dim) or (patch_pipe_root_out_dim < patch_pipe_tree_out_dim) :
            raise ValueError("The following must be true: embeddim > patch_pipe_root_out_dim > patch_pipe_tree_out_dim")

        if (embed_dim < patch_pipe_root_out_dim) or (patch_pipe_root_out_dim < patch_pipe_tree_out_dim) :
            raise ValueError("The following must be true: embeddim > patch_pipe_root_out_dim > patch_pipe_tree_out_dim")

        self.device = device
        self.id = id
        self.img_size = img_size
        self.channels = channels
        self.patch_size = patch_size
        self.classes = classes
        self.embed_dim = embed_dim
        self.patch_pipe_forks = patch_pipe_root_forks
        self.patch_pipe_merges = patch_pipe_tree_merges

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

        self.patch_embed = PatchEmbedding(embed_config)

        self.root = Root(embed_dim,
                        patch_pipe_root_forks,
                        patch_pipe_root_depth,
                        patch_pipe_root_out_dim)
        
        self.tree = Tree(patch_pipe_root_out_dim,
                        patch_pipe_tree_merges,
                        patch_pipe_tree_depth,
                        patch_pipe_tree_out_dim,
                        prod(patch_pipe_root_forks),
                        self.patches_pr_img
                        )
        
        self.classifier = nn.Linear(patch_pipe_tree_out_dim, classes)
        
        self.add_module('patch_embed', self.patch_embed)
        self.add_module('root', self.root)
        self.add_module('tree', self.tree)


    def forward(self, x):
        x = self.batch_norm(x)
        x = self.patch_embed(x)
        x = self.root(x)
        x = self.tree(x)
        x = self.classifier(x)
        return x





# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")



# model = tit()
# print(model.device)
# s = 0
# for n,param in model.named_parameters():
#     s+= param.numel()
#     #print(n)

# print("params: ",s)
# # dummy img batch: 32x3x124x124
# dummy_img = torch.rand(32,3,128,128)

# r = model(dummy_img)

# print(len(r))
# print(r[0].shape)
# print(r[len(r)-1].shape)

# # print names of params
# # for n,_ in model.named_parameters():
# #     print(n)