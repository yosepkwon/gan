import os
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset

# ------------------------- CelebA Dataset Class -------------------------
class CustomCelebA(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        CelebA 데이터셋을 로드하고 속성 라벨을 구성하는 클래스

        Parameters:
            root (str): 전체 CelebA 폴더의 루트 경로
            split (str): 데이터 분할 정보 ('train' 또는 'val')
            transform (callable): 이미지 전처리 transform (e.g., Resize, ToTensor)
        """
        # 이미지 및 속성 파일 경로 설정
        self.image_dir = os.path.join(root, 'celeba', 'img_align_celeba')
        self.attr_path = os.path.join(root, 'celeba', 'Anno', 'list_attr_celeba.txt')
        self.transform = transform
        self.selected_attrs = ['Blond_Hair', 'Male', 'Young']  # 사용할 속성 3가지

        # 속성 파일 읽기
        lines = open(self.attr_path, 'r').readlines()
        self.filenames = []  # 이미지 파일명을 저장
        self.labels = []     # 선택된 속성들의 0/1 벡터 저장

        # 속성 이름 → 인덱스 딕셔너리 생성 (2번째 줄 기준)
        self.attr2idx = {attr_name: idx for idx, attr_name in enumerate(lines[1].split())}

        # 각 이미지에 대해 선택 속성값만 추출하여 0/1로 정규화해 저장
        for line in lines[2:]:
            split = line.strip().split()
            filename = split[0]
            values = split[1:]
            label = [(int(values[self.attr2idx[attr]]) + 1) // 2 for attr in self.selected_attrs]
            self.filenames.append(filename)
            self.labels.append(label)

    def __getitem__(self, index):
        """
        인덱스에 해당하는 이미지와 라벨을 반환

        Returns:
            image (Tensor): 전처리된 이미지
            label (Tensor): 선택된 속성들의 0/1 벡터 (FloatTensor)
        """
        filename = self.filenames[index]
        label = torch.FloatTensor(self.labels[index])  # 라벨을 float32 텐서로 변환

        # 이미지 로드
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path)

        # 전처리 transform 적용
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        # 전체 샘플 개수 반환
        return len(self.filenames)


# ------------------------- RaFD Dataset Class -------------------------
class RafdDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        RaFD 데이터셋을 로드하고 감정 레이블을 처리하는 클래스

        Parameters:
            root (str): RaFD 데이터셋 루트 경로
            transform (callable): 이미지 전처리 transform
        """
        # 감정 레이블이 있는 CSV 파일과 이미지 루트 경로 지정
        self.label_path = os.path.join(root, 'train_labels.csv')
        self.root = os.path.join(root, 'DATASET', 'train')
        self.transform = transform

        # CSV 파일 읽어서 이미지 파일명과 레이블 추출
        df = pd.read_csv(self.label_path)
        self.images = df['image'].tolist()  # 파일명 리스트
        self.labels = df['label'].tolist()  # 정수형 감정 레이블 리스트
        self.num_classes = 8  # 감정 클래스 수 (RaFD 기준)

    def _one_hot(self, label):
        """
        정수형 레이블을 원-핫 벡터로 변환

        Parameters:
            label (int): 감정 클래스 인덱스

        Returns:
            list[int]: 원-핫 인코딩된 벡터
        """
        vec = [0] * self.num_classes
        vec[label] = 1
        return vec

    def __getitem__(self, index):
        """
        인덱스에 해당하는 이미지와 라벨 반환

        Returns:
            image (Tensor): 전처리된 이미지
            label (Tensor): 원-핫 벡터로 인코딩된 감정 레이블
        """
        img_name = self.images[index]
        label = self._one_hot(self.labels[index])  # 정수 → 원-핫

        # 예: train/2/image_001.jpg 와 같은 경로 구성
        image_path = os.path.join(self.root, str(self.labels[index]), img_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label)

    def __len__(self):
        # 전체 샘플 수 반환
        return len(self.images)
