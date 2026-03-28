import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import os
import numpy as np
from sklearn.model_selection import KFold # 引入 KFold

# ==========================================
# 1. 核心模块：CBAM 双重注意力 (空间+通道)
# (此部分与您提供的代码一致，无需修改)
# ==========================================
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel_attn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_weight

# ==========================================
# 2. 核心模块：多头加性注意力
# (此部分与您提供的代码一致，无需修改)
# ==========================================
class MultiHeadAdditiveAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAdditiveAttention, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, lstm_out):
        head_outputs = []
        for attn_net in self.heads:
            scores = attn_net(lstm_out)
            weights = F.softmax(scores, dim=1)
            context = torch.sum(weights * lstm_out, dim=1)
            head_outputs.append(context)
        return torch.cat(head_outputs, dim=1)

# ==========================================
# 3. 完整模型架构
# (此部分与您提供的代码一致，无需修改)
# ==========================================
class UltimateSTFTClassifier(nn.Module):
    def __init__(self, num_classes=3, num_heads=4):
        super(UltimateSTFTClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(4, 1))
        )
        self.cbam = CBAM(in_channels=64)
        self.hidden_dim = 128
        self.lstm = nn.LSTM(input_size=64 * 32, hidden_size=self.hidden_dim,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.multihead_attn = MultiHeadAdditiveAttention(hidden_dim=self.hidden_dim * 2, num_heads=num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 * num_heads, 128),
            nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.cbam(x)
        batch, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, t, -1)
        lstm_out, _ = self.lstm(x)
        attn_out = self.multihead_attn(lstm_out)
        return self.classifier(attn_out)

# ==========================================
# 4. 数据加载与训练主程序 (核心修改区域)
# ==========================================
def main():
    # --- 超参数配置 ---
    DATA_PATH = os.path.join(os.path.dirname(__file__), "pic") if "__file__" in dir() else "./pic"
    IMG_SIZE = 256
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 2e-4
    K_FOLDS = 5 # 定义交叉验证折数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 图像预处理 ---
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # --- 2. 加载完整数据集 ---
    if not os.path.exists(DATA_PATH):
        print(f"错误：找不到文件夹 {DATA_PATH}，请确保路径正确。")
        return
    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    print(f"成功加载数据集，共 {len(full_dataset)} 张图像，类别映射：{full_dataset.class_to_idx}")

    # --- 3. K-Fold 交叉验证设置 ---
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    # K-Fold 交叉验证循环
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f'{"="*20} FOLD {fold+1}/{K_FOLDS} {"="*20}')

        # 使用 SubsetRandomSampler 定义训练集和验证集的数据加载器
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, num_workers=2)
        val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_subsampler, num_workers=2)

        # 在每一折开始时，重新实例化模型、优化器和调度器
        model = UltimateSTFTClassifier(num_classes=len(full_dataset.classes)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        best_val_acc = 0.0

        # --- 4. 单次 Fold 的训练循环 ---
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # 新增：计算训练集准确率
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # 验证
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            scheduler.step()
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            # 保存当前 Fold 中最好的模型
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), f"stft_model_fold_{fold+1}.pth")


            print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
        
        print(f"Fold {fold+1} Best Val Acc: {best_val_acc:.2f}%")
        fold_results.append(best_val_acc)

    # --- 5. 打印最终结果 ---
    print(f'{"="*20} K-FOLD CROSS VALIDATION RESULTS {"="*20}')
    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print(f"所有折的验证准确率: {fold_results}")
    print(f"平均验证准确率: {avg_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"模型已保存为 stft_model_fold_1.pth, stft_model_fold_2.pth ...")


if __name__ == "__main__":
    main()
