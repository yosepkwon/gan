import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn

# Residual Block
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

# Generator
class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=3, repeat_num=6):
        super(Generator, self).__init__()
        
        layers = []
        # 첫 번째 Convolution (입력은 RGB 3채널 + 속성 c_dim)
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3))
        layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(curr_dim*2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim *= 2

        # Bottleneck Residual Blocks
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim=curr_dim))

        # Up-sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1))
            layers.append(nn.InstanceNorm2d(curr_dim//2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim //= 2

        # 마지막 Convolution (RGB 3채널 복구)
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # c: (batch, c_dim) -> (batch, c_dim, 1, 1) -> (batch, c_dim, H, W)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)  # 채널 방향으로 합치기
        return self.main(x)



# Generator 모델 로드 (G)
G = Generator(conv_dim=64, c_dim=3, repeat_num=6).cuda()

# 학습된 모델 불러오기
G.load_state_dict(torch.load('generator.pth'))  # 저장된 모델 경로
G.eval()  # 평가 모드로 전환

# 테스트할 이미지 불러오기
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 원본 이미지 불러오기
img_path = r'C:\Users\PRO\Desktop\3rd\opensource\celeba\img_align_celeba_aligned\aligned_005304.jpg'  # 예시 파일
image = Image.open(img_path).convert('RGB')
image = transform(image).unsqueeze(0).cuda()  # (1, 3, 128, 128)

# 3개 속성 변환할 준비 (Black_Hair, Blond_Hair, Brown_Hair, Male, Young)
# 원하는 target 속성 설정
attr_list = [
    [0, 1, 1],  
    [1, 0, 0],
    [0, 0, 1]
]

# 변환된 이미지 저장할 리스트
fake_images = []

for attr in attr_list:
    attr = torch.FloatTensor(attr).unsqueeze(0).cuda()  # (1, 5)
    fake_image = G(image, attr)  # 변환
    fake_images.append(fake_image)

# 원본 + 변환된 이미지 모두 합치기
images_concat = torch.cat([image] + fake_images, dim=0)

# [-1,1] 범위를 [0,1]로 변환해서 저장
images_concat = (images_concat + 1) / 2

# 폴더 없으면 만들기
os.makedirs('test_results', exist_ok=True)

# 저장
vutils.save_image(images_concat, 'test_results/result1.png', nrow=len(attr_list)+1)

print('변환된 이미지가 test_results/result.png에 저장되었습니다.')
