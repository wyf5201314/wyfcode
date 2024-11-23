import torch
from torch import nn, einsum
import torch.nn.functional as F
import random

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):  # Norm层
    def __init__(self, dim, fn):  # dim是维度
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):  # √
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        dropout = 0.1
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):  # √
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.4):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):  # √
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 提取center patch
class ExtractCenter:
    def __init__(self, patch_size, center_size):
        self.patch_size = patch_size
        self.center_size = center_size
    def get_centersize(self):
        return self.center_size
    def extract_center(self, input):
        """
        Extract the center 5x5 region for each pixel in a 13x13 neighborhood.

        Args:
        - input_data: Tensor of shape (a, 64, 13, 13)

        Returns:
        - center: Tensor of shape (a, 64, 5, 5)
        """
        half_patch = self.patch_size // 2
        half_center = self.center_size // 2

        # Extract center region
        center = input[:, :, half_patch - half_center:half_patch + half_center + 1,
                            half_patch - half_center:half_patch + half_center + 1]
        # print('ExtractCenter_shape:',center.shape)
        return center

# transformer encoder 特征提取模块
class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.4):
        super().__init__()
        self.layers = nn.ModuleList([])
        dim = 512
        for _ in range(1):  # transformer block数
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=0.4)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Center_Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.4):
        super().__init__()
        self.layers = nn.ModuleList([])
        dim = 512
        for _ in range(1):  # transformer block数
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=0.4)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, dim):
        super(PositionalEncoding, self).__init__()
        self.dim = dim
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

    def forward(self, center_coords):
        b, num, p_square = self.positional_embedding.shape
        distances = self.calculate_distances(center_coords, p_square)
        return self.positional_embedding * distances.unsqueeze(0)
    def calculate_distances(self, center_coords, p_square):
        patches_per_side = int(p_square ** 0.5)
        distances = torch.zeros(p_square)
        for i in range(patches_per_side):
            for j in range(patches_per_side):
                distances[i * patches_per_side + j] = self.calculate_distance(center_coords, i + 1, j + 1)
        return distances
    def calculate_distance(self, center_coords, i, j):
        return 1.0 / (torch.sqrt((center_coords[0] - i) ** 2 + (center_coords[1] - j) ** 2) + 1)


class ViT(nn.Module): # 加cls token 与patch embedding
    def __init__(self, image_size, num_classes, dim, heads, mlp_dim, pool='cls', channels=3, dim_head=64,
                 dropout=0., emb_dropout=0.1):
        super(ViT, self).__init__()

        # assert  image_height % patch_height ==0 and image_width % patch_width == 0

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        patch_dim = image_size ** 2
        num_patches = dim_head
        assert pool in {'cls', 'mean'}

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=image_height, p2=image_width),
        #     nn.Linear(patch_dim, dim)
        # )
        # self.pointwise = nn.Conv2d(64, 64, kernel_size=1)

        self.position_importance = PositionalEncoding(num_patches,patch_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Transformer(dim, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        b, d, p, p = x.shape
        x = x.reshape(b, d, -1)
        # x = x + self.position_importance(x)

        x = self.to_patch_embedding(x)  # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape  # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d',
                            b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # 将cls_token拼接到patch token中去       (b, 65, dim)

        # x = self.position_importance(x)
        x = x + self.pos_embedding[:, :(n + 1)]  # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)

        # x = self.transformer(x)  # (b, 65, dim)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # (b, dim)
        #
        # x = self.to_latent(x)  # Identity (b, dim)
        # print(x.shape)

        return x  # (b, 65, num_classes)

# msssfblock
class msssfblock(nn.Module):
    def __init__(self,
                 dim,
                 center_dim,
                 depth,
                 heads,
                 mlp_dim,
                 num_classes,
                 image_size,
                 center_size,
                 dim_head):
        super().__init__()

        # branch 1

        self.image_size = image_size
        patch_dim = image_size ** 2
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.dim = dim
        self.num_classes = num_classes
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.extractor = ExtractCenter(image_size,center_size)

        self.transformer_encoder = Transformer(self, dim, heads, dim_head, mlp_dim)
        self.center_encoder = Center_Transformer(self, dim, heads, dim_head, mlp_dim)


        self.catdim = center_dim + dim  # 多尺度特征融合

    # Multi-Scale Embedding Exchange
    def msee(self, output1, output2):
        a = random.randint(1, 9)
        class_token1 = output1[a, :].clone()  # 创建 class_token1 的副本
        class_token2 = output2[a, :].clone()  # 创建 class_token2 的副本

        # branch1/2 exchange class token
        output1[a, :] = F.normalize(class_token2, dim=-1)
        output2[a, :] = F.normalize(class_token1, dim=-1)

        return output1, output2

    def cls_swap(self, output1, output2):
        class_token1 = output1[:, 0, :].clone()  # 创建 class_token1 的副本
        class_token2 = output2[:, 0, :].clone()  # 创建 class_token2 的副本

        # if class_token1.size(0) != class_token2.size(0):
        #     class_token1 = torch.matmul(self.matrix_1to2, class_token1)
        #     class_token2 = torch.matmul(self.matrix_2to1, class_token2)

        # branch1/2 exchange class token
        output1[:, 0, :] = F.normalize(class_token2, dim=-1)
        output2[:, 0, :] = F.normalize(class_token1, dim=-1)

        return output1, output2

    def forward(self, input1,input2):

        output1 = self.transformer_encoder(input1)
        output2 = self.center_encoder(input2)

        # class token交换模块
        output1,output2 = self.cls_swap(output1,output2)


        return output1,output2

# 总网络
class MSSSF(nn.Module):
    def __init__(self,
                 dim,
                 center_dim,
                 depth,
                 heads,
                 mlp_dim,
                 num_classes,
                 image_size,
                 center_size,
                 dim_head,
                 pool='cls'):
        super().__init__()

        # Transformer branch

        self.image_size = image_size
        patch_dim = image_size ** 2
        self.dim = dim
        self.num_classes = num_classes
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.dim_head = dim_head
        self.pool = pool
        self.to_latent = nn.Identity()
        self.catdim = dim + dim
        self.linear = nn.Linear(self.catdim, dim)

        self.extractor = ExtractCenter(image_size,center_size)
        self.transformer_branch = ViT(image_size, num_classes, dim, heads, mlp_dim)
        self.center_branch = ViT(center_size, num_classes, dim, heads, mlp_dim)

        self.msssf = msssfblock(
                 dim,
                 center_dim,
                 depth,
                 heads,
                 mlp_dim,
                 num_classes,
                 image_size,
                 center_size,
                 dim_head)

        self.layers = nn.ModuleList([
            msssfblock(
                dim,
                center_dim,
                depth,
                heads,
                mlp_dim,
                num_classes,
                image_size,
                center_size,
                dim_head
            ) for _ in range(depth)
        ])
        # self.skipcat = nn.ModuleList([])
        # for _ in range(depth - 2):
        #     self.skipcat.append(
        #         nn.Conv2d(num_channel, num_channel, [1, 2], 1, 0)
        #     )

        self.fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.catdim),
            nn.Linear(self.catdim, self.num_classes)
        )
    # def swap(self,class_token1,class_token2,output1,output2):
    #     output1[ 0, :] = F.normalize(class_token2, dim=-1)
    #     output2[ 0, :] = F.normalize(class_token1, dim=-1)

    def forward(self, input):
        center_input = self.extractor.extract_center(input)  # get center patch

        input1 = self.transformer_branch(input)
        input2 = self.center_branch(center_input)

        for layer in self.layers:
            output1,output2 = layer(input1,input2)

        output = torch.cat((output1,output2),dim = -1)

        output_2d = output.mean(dim=1) if self.pool == 'mean' else output[:, 0]  # (b, dim)
        output_2d = self.to_latent(output_2d)  # Identity (b, catdim)
        # print("output shape:" + output.shape)

        # Fully connected layer
        # output = self.fc(output)
        # output = self.to_latent(output)
        output_2d = self.mlp_head(output_2d)

        return output_2d