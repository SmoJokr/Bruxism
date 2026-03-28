import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import timm

# ==========================================
# 1. 模型定义：基于 Swin Transformer V2
# ==========================================
class SwinV2STFTClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(SwinV2STFTClassifier, self).__init__()

        # 如果您的 STFT 图是彩色热力图，强烈建议用 3 通道 (in_chans=3)
        # 如果是纯黑白矩阵，请改为 in_chans=1
        self.model = timm.create_model(
            'swinv2_tiny_window8_256',
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3  # 默认使用3通道，充分利用预训练权重
        )

    def forward(self, x):
        return self.model(x)


# ==========================================
# 2. 数据处理与训练流程
# ==========================================
def main():
    # --- 超参数配置 ---
    DATA_PATH = "./pic"
    IMG_SIZE = 256  
    BATCH_SIZE = 16
    EPOCHS = 50
    WARMUP_EPOCHS = 5  # 预热 Epoch 数 (Transformer 必备)
    LR = 1e-4  
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 基础预处理 (无数据增强) ---
    transform = transforms.Compose([
        # 如果上方 in_chans=1，请取消注释下面这行，并将 Normalize 改为 [0.5], [0.5]
        # transforms.Grayscale(1), 
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # --- 2. 加载数据集 ---
    if not os.path.exists(DATA_PATH):
        print(f"路径不存在: {DATA_PATH}")
        return

    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    print(f"数据集加载成功，总计 {len(full_dataset)} 张图片，类别映射: {full_dataset.class_to_idx}")

    # 划分训练集和验证集 (8:2)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 3. 模型与优化器 ---
    model = SwinV2STFTClassifier(num_classes=len(full_dataset.classes), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    
    # 学习率调度：前 5 个 Epoch 线性 Warmup，之后余弦退火
    scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[WARMUP_EPOCHS])

    # --- 4. 训练循环 ---
    print(f"开始在 {DEVICE} 上训练 Swin Transformer V2 (无数据增强版本)...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        # -- 训练阶段 --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 统计训练指标
            train_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total

        # -- 验证阶段 --
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                # 统计验证指标
                val_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 打印当前 Epoch 结果
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] LR: {current_lr:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_swinv2_stft.pth")

    print(f"训练完成！最高验证集准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
