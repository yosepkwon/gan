# model.py
# import torch
# import torch.nn as nn
# import numpy as np

# class ResidualBlock(nn.Module):
#     def __init__(self, dim):
#         super(ResidualBlock, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1),
#             nn.InstanceNorm2d(dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim, dim, 3, 1, 1),
#             nn.InstanceNorm2d(dim)
#         )

#     def forward(self, x):
#         return x + self.main(x)

# class Generator(nn.Module):
#     def __init__(self, conv_dim=64, c_dim=3, repeat_num=6):
#         super(Generator, self).__init__()
#         layers = [
#             nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3),
#             nn.InstanceNorm2d(conv_dim),
#             nn.ReLU(inplace=True)
#         ]
#         curr_dim = conv_dim
#         for _ in range(2):
#             layers += [
#                 nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1),
#                 nn.InstanceNorm2d(curr_dim*2),
#                 nn.ReLU(inplace=True)
#             ]
#             curr_dim *= 2
#         for _ in range(repeat_num):
#             layers.append(ResidualBlock(curr_dim))
#         for _ in range(2):
#             layers += [
#                 nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1),
#                 nn.InstanceNorm2d(curr_dim//2),
#                 nn.ReLU(inplace=True)
#             ]
#             curr_dim //= 2
#         layers.append(nn.Conv2d(curr_dim, 3, 7, 1, 3))
#         layers.append(nn.Tanh())
#         self.main = nn.Sequential(*layers)

#     def forward(self, x, c):
#         c = c.view(c.size(0), c.size(1), 1, 1)
#         c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
#         x = torch.cat([x, c], dim=1)
#         return self.main(x)

# class Discriminator(nn.Module):
#     def __init__(self, image_size=128, conv_dim=64, c_dim=3, repeat_num=6):
#         super(Discriminator, self).__init__()
#         layers = [
#             nn.Conv2d(3, conv_dim, 4, 2, 1),
#             nn.LeakyReLU(0.01)
#         ]
#         curr_dim = conv_dim
#         for _ in range(1, repeat_num):
#             layers += [
#                 nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1),
#                 nn.LeakyReLU(0.01)
#             ]
#             curr_dim *= 2
#         kernel_size = int(image_size / (2 ** repeat_num))
#         self.main = nn.Sequential(*layers)
#         self.conv1 = nn.Conv2d(curr_dim, 1, 3, 1, 1)
#         self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

#     def forward(self, x):
#         h = self.main(x)
#         out_src = self.conv1(h)
#         out_cls = self.conv2(h)
#         out_cls = out_cls.view(out_cls.size(0), -1)
#         return out_src, out_cls
# model.py
import torch
import torch.nn as nn
import numpy as np

# -------------------- Generator --------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=3, repeat_num=6):
        super(Generator, self).__init__()
        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True)
        ]
        curr_dim = conv_dim

        # 다운샘플링
        for _ in range(2):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr_dim))

        # 업샘플링
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim //= 2

        # 출력층
        layers.append(nn.Conv2d(curr_dim, 3, 7, 1, 3))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # 라벨 벡터 c를 spatial 차원으로 broadcast
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

# -------------------- Discriminator --------------------
class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, use_dropout=False):
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01)
        ]

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            next_dim = curr_dim * 2
            layers.append(nn.Conv2d(curr_dim, next_dim, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.01))
            if use_dropout:
                layers.append(nn.Dropout(0.3))  # Dropout 추가
            curr_dim = next_dim

        kernel_size = int(image_size / np.power(2, repeat_num))  # 최종 출력 크기
        self.main = nn.Sequential(*layers)
        self.conv_src = nn.Conv2d(curr_dim, 1, 3, 1, 1)  # 진짜/가짜 판단
        self.conv_cls = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)  # 감정 분류

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_src(h)
        out_cls = self.conv_cls(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

