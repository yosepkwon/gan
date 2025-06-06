import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import cv2
import numpy as np

# ---------------- Residual Block 정의 ----------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        """
        Generator 내부에서 사용되는 Residual Block
        :param dim: 입력과 출력 feature map의 채널 수
        """
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),            # 3x3 Convolution
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.main(x)  # 입력과 출력의 skip connection (residual)

# ---------------- Generator 정의 ----------------
class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=3, repeat_num=6):
        """
        StarGAN Generator 구조 정의
        :param conv_dim: 초기 필터 수
        :param c_dim: 속성 조건 차원 수 (예: Blond_Hair, Male, Young)
        :param repeat_num: Residual block 반복 횟수
        """
        super(Generator, self).__init__()
        layers = []

        # 초기 Convolution: 이미지 + 속성 벡터 결합
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling: 2회 수행 (해상도 감소, feature 수 증가)
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(curr_dim*2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim *= 2

        # Bottleneck: Residual Block 반복
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim=curr_dim))

        # Up-sampling: 2회 수행 (해상도 복원)
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(curr_dim//2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim //= 2

        # 출력 Convolution: RGB 이미지 재구성
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3))
        layers.append(nn.Tanh())  # 출력값을 [-1, 1]로 정규화

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        """
        Generator의 forward 연산 정의
        :param x: 입력 이미지 (B, 3, H, W)
        :param c: 조건 벡터 (B, c_dim)
        """
        c = c.view(c.size(0), c.size(1), 1, 1)  # 조건 벡터를 4D로 reshape
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))  # 이미지 크기에 맞게 확장
        x = torch.cat([x, c], dim=1)  # 이미지와 조건 결합
        return self.main(x)

# ---------------- Generator 로드 및 평가모드 설정 ----------------
G = Generator(conv_dim=64, c_dim=3, repeat_num=6).cuda()
G.load_state_dict(torch.load('generator.pth'))  # 학습된 가중치 로드
G.eval()  # 평가 모드 설정 (Dropout/BatchNorm 비활성화)

# ---------------- 입력 이미지 전처리 ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),                     # 128x128 크기로 resize
    transforms.ToTensor(),                             # Tensor로 변환 (0~1)
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)     # [-1, 1] 정규화
])

# 테스트할 이미지 로드
img_path = r'C:\Users\PRO\Desktop\3rd\opensource\celeba\img_align_celeba\000243.jpg'
image = Image.open(img_path).convert('RGB')           # PIL 이미지 불러오기
image = transform(image).unsqueeze(0).cuda()          # (1, 3, 128, 128)

# ---------------- 조건 속성 설정 ----------------
attr_list = [
    [0, 1, 1],  # 예: 금발X, 남성O, 젊음O
    [1, 0, 0],  # 예: 금발O, 여성O, 젊음X
    [1, 1, 1],  # 예: 금발O, 남성O, 젊음O
]

# ---------------- 이미지 변환 실행 ----------------
fake_images = []
for attr in attr_list:
    attr = torch.FloatTensor(attr).unsqueeze(0).cuda()  # (1, 3)
    fake_image = G(image, attr)
    fake_images.append(fake_image)

# 원본 + 변환 이미지 하나로 결합
images_concat = torch.cat([image] + fake_images, dim=0)  # (B+1, 3, H, W)
images_concat = (images_concat + 1) / 2  # [-1,1] → [0,1] 범위로 변환

# ---------------- 결과 이미지 저장 ----------------
os.makedirs('test_results', exist_ok=True)
result_path = 'test_results/result1.png'
vutils.save_image(images_concat, result_path, nrow=len(attr_list)+1)
print(f'변환된 이미지가 {result_path}에 저장되었습니다.')

# ---------------- OpenCV로 이미지 출력 ----------------
img_cv = cv2.imread(result_path)  # OpenCV는 BGR로 읽기
if img_cv is not None:
    cv2.imshow('Transformed Results', img_cv)  # 윈도우에 출력
    print("아무 키나 누르면 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("이미지를 OpenCV로 불러오는 데 실패했습니다.")
