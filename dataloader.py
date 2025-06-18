import os
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset

# CelebA Dataset Class
class CustomCelebA(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        CelebA 데이터셋을 로드하고 속성 라벨을 구성하는 클래스

        root (): 전체 CelebA 폴더의 루트 경로
        split (): 데이터 분할 정보
        transform (): 이미지 전처리
        """
        # 경로 설정
        self.img_dir = os.path.join(root, 'celeba', 'img_align_celeba')
        self.attr_path = os.path.join(root, 'celeba', 'Anno', 'list_attr_celeba.txt')
        self.transform = transform
        self.attrs = ['Blond_Hair', 'Male', 'Young']  # 속성 

        # 속성 파일 읽기
        lines = open(self.attr_path, 'r').readlines()
        self.filenames = []  # 이미지 파일명 저장
        self.labels = []     # 속성들의 0/1 벡터 저장

        # 속성 이름 → 인덱스 딕셔너리 생성
        self.attr2idx = {attr_name: idx for idx, attr_name in enumerate(lines[1].split())}

        # 속성값을 0/1로 정규화해 저장
        for line in lines[2:]:
            split = line.strip().split()
            filename = split[0]
            values = split[1:]
            label = [(int(values[self.attr2idx[attr]]) + 1) // 2 for attr in self.attrs]
            self.filenames.append(filename)
            self.labels.append(label)

    def __getitem__(self, index):
        """
        인덱스에 해당하는 이미지와 라벨을 반환

        image (): 전처리된 이미지
        label (): 선택된 속성들의 0/1 벡터 (FloatTensor)
        """
        filename = self.filenames[index]
        label = torch.FloatTensor(self.labels[index]) 

        # 이미지 로드
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)

        # 전처리 transform 적용
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        # 전체 샘플 개수 반환
        return len(self.filenames)
