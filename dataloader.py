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


# RaFD Dataset Class
class RafdDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        RaFD 데이터셋을 로드하고 감정 레이블을 처리하는 클래스

        root (): RaFD 데이터셋 루트 경로
        transform (): 이미지 전처리 
        """
        # 경로 지정
        self.label_path = os.path.join(root, 'train_labels.csv')
        self.root = os.path.join(root, 'DATASET', 'train')
        self.transform = transform

        # CSV 파일 읽어서 이미지 파일명과 레이블 추출
        df = pd.read_csv(self.label_path)
        self.images = df['image'].tolist()  # 파일명 리스트
        self.labels = df['label'].tolist()  # 레이블 리스트
        self.num_classes = 8 

    def _one_hot(self, label):
        """
        정수형 레이블을 원-핫 벡터로 변환

        label (): 감정 클래스 인덱스
        list[]: 원-핫 인코딩된 벡터
        """
        vec = [0] * self.num_classes
        vec[label] = 1
        return vec

    def __getitem__(self, index):
        """
        인덱스에 해당하는 이미지와 라벨 반환

        image (): 전처리된 이미지
        label (): 원-핫 벡터로 인코딩된 감정 레이블
        """
        img_name = self.images[index]
        label = self._one_hot(self.labels[index])  # 정수 → 원-핫

        image_path = os.path.join(self.root, str(self.labels[index]), img_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label)

    def __len__(self):
        # 전체 샘플 수 반환
        return len(self.images)
