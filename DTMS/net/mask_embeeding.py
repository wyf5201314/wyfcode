import torch 
import torch.nn as nn 
import random 
from torch.cuda.amp import autocast as autocast
# from main_temp import device

from torchvision.transforms import transforms
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



def ShuffleIndex(index: list, sample_ratio: float):
    '''
    这个函数用于从给定的索引列表中随机选择一定比例的索引，并返回被选择的索引列表以及剩余的索引列表。它接受两个参数：index（索引列表）和 sample_ratio（采样比率）。
函数首先检查索引列表长度是否大于4，如果小于4会抛出一个 ValueError。然后，根据给定的采样比率，计算需要保留的索引数量。
接着，函数创建一个临时的索引列表副本，并在副本中随机选择索引，直到剩余索引数量达到设定的数量。被选中的索引会放入 sample_list 中，剩余的索引会放入 mask_list 中。
最后，函数返回 sample_list 和 mask_list。
    '''
    sample_list = []
    if len(index) < 4:
        raise ValueError("ipnuts must be more than 4")
    else:
        remain_length = int((1 - sample_ratio) * len(index))
        temp_index = index.copy()
        while len(temp_index) > remain_length:
            sample = random.choice(temp_index) # 被选中的索引
            sample_list.append(sample)
            temp_index.remove(sample)
        
        mask_list = [x for x in index if x not in sample_list]  # get the remain index not in cls token and not in sample_index # 被mask的索引
        # assert len(sample_list) == int(len(index) * sample_ratio), "sample length must be same as the ratio!!!"
    return sample_list, mask_list 


def MaskEmbeeding(token_emb, mask_ratio):
    """get the mask embeeding after patch_emb + pos_emb
    这个函数用于对输入的 token embedding 进行掩码操作。接受两个参数：token_emb（token embedding 张量）和 mask_ratio（掩码比率）。
函数首先根据输入张量的长度生成索引列表。然后，调用 ShuffleIndex 函数根据给定的掩码比率从索引列表中选择一定比例的索引作为样本索引，剩余的作为掩码索引。
接着，函数根据样本索引从输入张量中选择对应的片段，并返回掩码后的张量、样本索引列表和掩码索引列表。
    """
    token_length = token_emb.shape[1]
    token_index = [x for x in range(1, token_length)]
    # print(len(token_index))
    mask_index, sample_index = ShuffleIndex(token_index, mask_ratio)
    token_sample = [0] + sample_index
    
    x = token_emb[:, token_sample, :]
    return x, sample_index, mask_index
        

class UnMaskEmbeeding_spa(nn.Module):
    """get the mask embeeding from the image -> 127 to embeeding, before the position embeeding
    用于将掩码后的张量进行解码操作，替换被掩码的元素为指定的值。
它们的 forward 方法接受三个参数：x（被掩码的张量）、sample_index（样本索引列表）和 mask_index（掩码索引列表）。
在 forward 方法中，首先将输入张量通过一个卷积层（UnMaskEmbeeding_spa）或全连接层（UnMaskEmbeeding_chan）进行处理，以生成对应的 patch embedding。
然后，根据给定的样本索引和掩码索引，将 patch embedding 中对应位置的值替换为掩码后的张量的值，并返回结果。
    """
    def __init__(self, input_size, embed_dim, in_chans, patch_size, num_patches,args):
        super().__init__()
        self.in_chans = 64
        self.embed_dim = embed_dim
        self.kernel_size = patch_size
        self.num_patches = num_patches
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        # used for mask images
        self.raw_inputs = torch.ones((in_chans, input_size, input_size))*127. / 255
        # self.raw_inputs = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(self.raw_inputs)
        self.raw_inputs = self.raw_inputs.unsqueeze(0)
        self.raw_inputs = self.raw_inputs.to(self.device)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)#.to(torch.device("cuda:1"))
        
    def forward(self, x, sample_index, mask_index):
        
        b, _, _ = x.shape
        raw_inputs = self.raw_inputs.expand(b, -1, -1, -1)
        decoder_embeeding = nn.Parameter(torch.zeros((b, 1 + self.num_patches, self.embed_dim))).to(self.device)
        # print(raw_inputs.device)
        embeeding = self.proj(raw_inputs) # b, c, h, w

        b, c, h, w = embeeding.shape
        patch_embeeding = embeeding.view(b, -1, c)[0, 0, :]



        decoder_embeeding[:, [0] + sample_index, :] = x
        decoder_embeeding[:, mask_index, :] = patch_embeeding
        
        return decoder_embeeding


class UnMaskEmbeeding_chan(nn.Module):
    """get the mask embeeding from the image -> 127 to embeeding, before the position embeeding
    """

    def __init__(self, input_size, embed_dim, in_chans, patch_size, num_patches,args):
        super().__init__()
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        # self.in_chans = in_chans
        self.in_chans = 64
        self.embed_dim = embed_dim
        self.kernel_size = patch_size
        self.num_patches = num_patches
        # used for mask images
        self.raw_inputs = torch.ones((64, input_size, input_size)) * 127. / 255
        # self.raw_inputs = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(self.raw_inputs)
        self.raw_inputs = self.raw_inputs.unsqueeze(0)
        self.raw_inputs = self.raw_inputs.to(self.device)
        self.proj = nn.Linear(input_size * input_size, embed_dim)

    def forward(self, x, sample_index, mask_index):
        b, _, _ = x.shape
        raw_inputs = self.raw_inputs.expand(b, -1, -1, -1)
        raw_inputs = raw_inputs.reshape(b,self.in_chans,-1)     #(b,64,625)
        decoder_embeeding = nn.Parameter(torch.zeros((b, 1 + self.num_patches, self.embed_dim))).to(self.device)    #(b,65,256)
        # print(raw_inputs.device)
        patch_embeeding = self.proj(raw_inputs)     #(b,64,256)

        # b, c, h, w = embeeding.shape
        # patch_embeeding = embeeding.view(b, c,-1)[0, 0, :]

        decoder_embeeding[:, [0] + sample_index, :] = x
        decoder_embeeding[:, mask_index, :] = patch_embeeding[0, 0, :]

        return decoder_embeeding

class UnMaskEmbeeding_center(nn.Module):
    """get the mask embeeding from the image -> 127 to embeeding, before the position embeeding
    """

    def __init__(self, input_size, embed_dim, in_chans, patch_size, num_patches,args):
        super().__init__()
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        # self.in_chans = in_chans
        self.in_chans = 64
        self.embed_dim = embed_dim
        self.kernel_size = patch_size
        self.num_patches = num_patches
        # used for mask images
        self.raw_inputs = torch.ones((64, input_size, input_size)) * 127. / 255
        # self.raw_inputs = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(self.raw_inputs)
        self.raw_inputs = self.raw_inputs.unsqueeze(0)
        self.raw_inputs = self.raw_inputs.to(self.device)
        self.proj = nn.Linear(input_size * input_size, embed_dim)

    def forward(self, x, sample_index, mask_index):
        b, _, _ = x.shape
        raw_inputs = self.raw_inputs.expand(b, -1, -1, -1)
        raw_inputs = raw_inputs.reshape(b,self.in_chans,-1)     #(b,64,625)
        decoder_embeeding = nn.Parameter(torch.zeros((b, 1 + self.num_patches, self.embed_dim))).to(self.device)    #(b,65,256)
        # print(raw_inputs.device)
        patch_embeeding = self.proj(raw_inputs)     #(b,64,256)

        # b, c, h, w = embeeding.shape
        # patch_embeeding = embeeding.view(b, c,-1)[0, 0, :]

        decoder_embeeding[:, [0] + sample_index, :] = x
        decoder_embeeding[:, mask_index, :] = patch_embeeding[0, 0, :]

        return decoder_embeeding

if __name__ == '__main__':
    a = [x for x in range(196)]
    sample = ShuffleIndex(a, 0.75)
    print(sample)
    
        
            