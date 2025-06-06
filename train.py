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
num_epochs = 50
c_dim = 3  # Blond_Hair, Male, Young
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [Early Stop]
patience = 5
best_loss = float('inf')
early_stop_counter = 0

# [Scheduler]
step_size = 10   # 에포크마다 감소
gamma = 0.5      # 감소 비율

def main():
    global best_loss, early_stop_counter

    # ----------------- 데이터 로더 -----------------
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    root = r'C:\Users\PRO\Desktop\3rd\opensource'
    train_dataset = CustomCelebA(root=root, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    # ----------------- 모델/Optimizer/Scheduler -----------------
    G = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6).to(device)
    D = Discriminator(image_size=image_size, conv_dim=64, c_dim=c_dim, repeat_num=6).to(device)

    g_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=step_size, gamma=gamma)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=step_size, gamma=gamma)

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
        epoch_g_loss = 0.0

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
            epoch_g_loss += g_loss.item()

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # -------- 로그 출력 --------
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

        print()
        torch.save(G.state_dict(), f'{checkpoint_dir}/generator_epoch{epoch+1}.pth')

        # -------- Early Stop 평가 --------
        avg_g_loss = epoch_g_loss / len(train_loader)
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"[Early Stop] {early_stop_counter}회 연속 성능 개선 없음 (Best G Loss: {best_loss:.4f})")
            if early_stop_counter >= patience:
                print(f"[Early Stop] 성능 향상 없음. 조기 종료합니다. (Epoch {epoch+1})")
                break

        # -------- Learning Rate Update --------
        g_scheduler.step()
        d_scheduler.step()

    # -------- 최종 저장 --------
    torch.save(G.state_dict(), 'generator.pth')
    print('학습 완료! generator.pth 저장되었습니다.')

if __name__ == '__main__':
    main()
