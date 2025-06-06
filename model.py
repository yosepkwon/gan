import torch
import torch.nn as nn
import numpy as np

# -------------------- Residual Block --------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, use_dropout=False):
        """
        Residual Block 정의
        - 입력 특징 맵에 작은 변화를 더해주는 skip connection 구조
        - 학습 안정성 및 gradient 흐름을 개선함
        - Dropout을 옵션으로 포함할 수 있음

        :param dim: 입력 채널 수 (= 출력 채널 수)
        :param use_dropout: 중간 Conv 사이에 Dropout 삽입 여부
        """
        super(ResidualBlock, self).__init__()
        layers = [
            nn.Conv2d(dim, dim, 3, 1, 1),        # 3x3 convolution (same padding)
            nn.InstanceNorm2d(dim),             # 정규화
            nn.ReLU(inplace=True),              # 비선형 활성화
            nn.Conv2d(dim, dim, 3, 1, 1),        # 두 번째 convolution
            nn.InstanceNorm2d(dim)              # 정규화
        ]
        if use_dropout:
            # 두 Conv 사이에 Dropout을 삽입하여 과적합 방지
            layers.insert(3, nn.Dropout(0.3))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # 입력과 출력의 합을 통해 residual 연결 수행
        return x + self.main(x)


# -------------------- Generator --------------------
class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=3, repeat_num=6, use_dropout=False):
        """
        Generator 네트워크 정의
        - 인풋 이미지와 속성 벡터를 받아 target domain의 이미지 생성
        - 다운샘플링 → Residual blocks → 업샘플링 구성
        - 속성 벡터를 채널 차원으로 결합함

        :param conv_dim: 초기 convolution filter 수
        :param c_dim: 속성 벡터 차원 수
        :param repeat_num: Residual block 수
        :param use_dropout: Residual block 내 dropout 여부
        """
        super(Generator, self).__init__()
        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3),    # 이미지(3채널) + 속성(c_dim) → conv
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True)
        ]
        curr_dim = conv_dim

        # ---- Down-sampling (2단계로 크기 절반씩 축소) ----
        for _ in range(2):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2

        # ---- Bottleneck: Residual Block 반복 ----
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr_dim, use_dropout=use_dropout))

        # ---- Up-sampling (2단계로 크기 2배씩 복원) ----
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim //= 2

        # ---- Output Layer (최종 RGB 이미지 생성) ----
        layers.append(nn.Conv2d(curr_dim, 3, 7, 1, 3))  # 3채널 출력
        layers.append(nn.Tanh())                       # [-1, 1] 정규화 출력

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        """
        Generator의 Forward 과정
        - 이미지 x와 속성 벡터 c를 concat하여 변환 결과 생성

        :param x: 입력 이미지 (B, 3, H, W)
        :param c: 속성 벡터 (B, c_dim)
        :return: 변환된 이미지 (B, 3, H, W)
        """
        c = c.view(c.size(0), c.size(1), 1, 1)              # (B, c_dim, 1, 1)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))  # (B, c_dim, H, W)
        x = torch.cat([x, c], dim=1)                        # 채널 방향으로 concat
        return self.main(x)


# -------------------- Discriminator --------------------
class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, use_dropout=False):
        """
        PatchGAN 기반 Discriminator
        - 이미지가 진짜/가짜인지 판별
        - 동시에 속성 분류도 수행 (멀티태스크)

        :param image_size: 입력 이미지 크기 (128x128 등)
        :param conv_dim: 시작 필터 수
        :param c_dim: 분류해야 할 속성 수
        :param repeat_num: Conv 층 반복 수
        :param use_dropout: 중간층 Dropout 여부
        """
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01)
        ]
        curr_dim = conv_dim

        # ---- Conv 블록 반복 (해상도 점차 축소) ----
        for i in range(1, repeat_num):
            next_dim = curr_dim * 2
            layers.append(nn.Conv2d(curr_dim, next_dim, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.01))
            if use_dropout:
                layers.append(nn.Dropout(0.3))  # 과적합 방지를 위한 Dropout 추가
            curr_dim = next_dim

        # 최종 feature map 크기 계산
        kernel_size = int(image_size / np.power(2, repeat_num))

        # 공유된 feature extractor
        self.main = nn.Sequential(*layers)

        # 진짜/가짜 판별 헤드
        self.conv_src = nn.Conv2d(curr_dim, 1, 3, 1, 1)

        # 속성 분류 헤드 (RaFD에서 사용됨)
        self.conv_cls = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        """
        Discriminator의 Forward 과정
        :param x: 입력 이미지 (B, 3, H, W)
        :return: (out_src: 진짜/가짜 판별 맵), (out_cls: 속성 분류 결과)
        """
        h = self.main(x)
        out_src = self.conv_src(h)  # 판별 결과 (진짜/가짜)
        out_cls = self.conv_cls(h)  # 속성 분류 결과

        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
