import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import CustomCelebA
from model import Generator, Discriminator

# ----------------- 하이퍼파라미터 -----------------
batch_size = 32
image_size = 128
num_epochs = 40
c_dim = 3  # Blond_Hair, Male, Young
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # ----------------- 데이터 로더 -----------------
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    root = r'C:\Users\PRO\Desktop\3rd\opensource'
    train_dataset = CustomCelebA(root=root, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    # ----------------- 모델/Optimizer/Loss -----------------
    G = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6).to(device)
    D = Discriminator(image_size=image_size, conv_dim=64, c_dim=c_dim, repeat_num=6).to(device)

    g_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    classification_loss = nn.BCEWithLogitsLoss()

    # ----------------- 체크포인트 로딩 -----------------
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_epoch = 0
    for f in os.listdir(checkpoint_dir):
        if f.startswith('generator_epoch') and f.endswith('.pth'):
            try:
                epoch_num = int(f.split('generator_epoch')[1].split('.pth')[0])
                resume_epoch = max(resume_epoch, epoch_num)
            except:
                continue

    checkpoint_path = os.path.join(checkpoint_dir, f'generator_epoch{resume_epoch}.pth')
    if resume_epoch > 0 and os.path.exists(checkpoint_path):
        G.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[✓] 체크포인트 {checkpoint_path} 로드 완료. Epoch {resume_epoch}부터 재시작합니다.")
    else:
        print("[!] 체크포인트 없음. 새로 학습을 시작합니다.")
        resume_epoch = 0

    # ----------------- 학습 루프 -----------------
    for epoch in range(resume_epoch, num_epochs):
        start_time = time.time()
        for i, (real_images, real_labels) in enumerate(train_loader):
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            # -------- Discriminator --------
            real_src, real_cls = D(real_images)
            d_loss_real = torch.mean((real_src - 1)**2)
            d_loss_cls = classification_loss(real_cls, real_labels)

            rand_idx = torch.randperm(real_labels.size(0))
            fake_labels = real_labels[rand_idx]
            fake_images = G(real_images, fake_labels)
            fake_src, _ = D(fake_images.detach())
            d_loss_fake = torch.mean(fake_src**2)

            d_loss = d_loss_real + d_loss_fake + d_loss_cls

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # -------- Generator --------
            fake_src, fake_cls = D(fake_images)
            g_loss_fake = torch.mean((fake_src - 1)**2)
            g_loss_cls = classification_loss(fake_cls, fake_labels)
            g_loss = g_loss_fake + g_loss_cls

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # -------- 로그 출력 (한 줄로 갱신) --------
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                steps_done = i + 1
                steps_total = len(train_loader)
                eta = (elapsed / steps_done) * (steps_total - steps_done)
                sys.stdout.write(
                    f"\rEpoch [{epoch+1}/{num_epochs}], "
                    f"Step [{i+1}/{steps_total}], "
                    f"D Loss: {d_loss.item():.4f}, "
                    f"G Loss: {g_loss.item():.4f}, "
                    f"ETA: {int(eta // 60)}m {int(eta % 60)}s"
                )
                sys.stdout.flush()

        print()  # Epoch 종료 후 줄바꿈
        torch.save(G.state_dict(), f'{checkpoint_dir}/generator_epoch{epoch+1}.pth')

    # -------- 최종 저장 --------
    torch.save(G.state_dict(), 'generator.pth')
    print('학습 완료! generator.pth 저장되었습니다.')

if __name__ == '__main__':
    main()
