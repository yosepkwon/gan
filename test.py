import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import cv2
import numpy as np

# Residual Block 정의
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        """
        Generator 내부에서 사용되는 Residual Block
        dim: 입력과 출력 feature map의 채널 수
        """
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

# Generator 정의
class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=3, repeat_num=6):
        """
        Generator 구조 정의
        conv_dim: 초기 필터 수
        c_dim: 속성 조건 차원 수 (예: Blond_Hair, Male, Young)
        repeat_num: Residual block 반복 횟수
        """
        super(Generator, self).__init__()
        layers = []

        # 초기 Convolution: 이미지 + 속성 벡터 결합
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling
        cur_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(cur_dim, cur_dim*2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(cur_dim*2))
            layers.append(nn.ReLU(inplace=True))
            cur_dim *= 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim=cur_dim))

        # Up-sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(cur_dim, cur_dim//2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(cur_dim//2))
            layers.append(nn.ReLU(inplace=True))
            cur_dim //= 2

        # 출력 Convolution
        layers.append(nn.Conv2d(cur_dim, 3, kernel_size=7, stride=1, padding=3))
        layers.append(nn.Tanh())  # 출력값을 [-1, 1]로 정규화

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        """
        forward 연산 정의
        x: 입력 이미지 (B, 3, H, W)
        c: 조건 벡터 (B, c_dim)
        """
        c = c.view(c.size(0), c.size(1), 1, 1)  
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))  # 이미지 크기에 맞게 확장
        x = torch.cat([x, c], dim=1)  # 이미지와 조건 결합
        return self.main(x)

# Generator 로드 및 평가모드 설정
G = Generator(conv_dim=64, c_dim=3, repeat_num=6).cuda()
G.load_state_dict(torch.load('generator.pth'))  # 학습된 가중치 로드
G.eval()  # 평가 모드 설정

# 입력 이미지 전처리 
transform = transforms.Compose([
    transforms.Resize((128, 128)),                     # 128x128 크기로 resize
    transforms.ToTensor(),                             # Tensor로 변환 (0~1)
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)    # [-1, 1] 정규화
])

# 테스트 이미지 로드
img_path = r'C:\Users\PRO\Desktop\3rd\opensource\celeba\img_align_celeba\000243.jpg'
image = Image.open(img_path).convert('RGB')           
image = transform(image).unsqueeze(0).cuda()          

# 속성 설정
attr_list = [
    [1, 0, 0],  # blond hair
    [0, 1, 0],  # gender
    [0, 0, 1],  # aged
    [1, 1, 1]   # mix
]

# 이미지 변환 실행
fake_images = []
for attr in attr_list:
    attr = torch.FloatTensor(attr).unsqueeze(0).cuda() 
    fake_image = G(image, attr)
    fake_images.append(fake_image)

# 원본 + 변환 이미지 하나로 결합
images_concat = torch.cat([image] + fake_images, dim=0)  
images_concat = (images_concat + 1) / 2  # [-1,1] → [0,1] 범위로 변환

# 결과 이미지 저장
os.makedirs('test_results', exist_ok=True)
result_path = 'test_results/result1.png'
vutils.save_image(images_concat, result_path, nrow=len(attr_list)+1)
print(f'변환된 이미지 {result_path}에 저장.')


# OpenCV로 이미지 출력

# 저장된 이미지 불러오기
img_cv = cv2.imread(result_path)

if img_cv is not None:
    # 이미지 크기 추출
    img_h, img_w, _ = img_cv.shape
    num_images = len(attr_list) + 1
    single_width = img_w // num_images

    # 폰트 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.6
    f_thickness = 1
    color = (0, 255, 0)  # Green

    # 상단 텍스트 삽입 
    labels = ['Original', 'Blond hair', 'Gender', 'Aged', 'Mix']
    for i in range(num_images):
        x = i * single_width + 5
        y = 20
        label = labels[i] if i < len(labels) else f'Attr{i}'
        cv2.putText(img_cv, label, (x, y), font, f_scale, 
                    color, f_thickness, cv2.LINE_AA)

    # 이미지 출력
    cv2.imshow('Results', img_cv)
    print("아무 키나 누르면 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("로드 실패.")
