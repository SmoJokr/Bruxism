import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import datasets, transforms
import os
import numpy as np
from sklearn.model_selection import KFold

# ==========================================
# 1. 核心模块：CBAM / MultiHead / Classifier
# (保持您的模型架构不变)
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
# 2. 新增：带路径的数据集包装器
# ==========================================
class ImageFolderWithPaths(Dataset):
    def __init__(self, root_dir, transform=None):
        self.inner_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        
    def __getitem__(self, index):
        data, label = self.inner_dataset[index]
        path = self.inner_dataset.samples[index][0] # 获取原始文件路径
        return data, label, path

    def __len__(self):
        return len(self.inner_dataset)

    @property
    def classes(self): return self.inner_dataset.classes
    @property
    def class_to_idx(self): return self.inner_dataset.class_to_idx

# ==========================================
# 3. 早停机制
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    DATA_PATH = "./pic"
    IMG_SIZE = 256
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 2e-4
    K_FOLDS = 5
    PATIENCE = 8
    HARD_SAMPLE_RATIO = 0.1  # 每折筛选出 Loss 最高的前 10% 样本
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 加载带路径的数据集
    full_dataset = ImageFolderWithPaths(DATA_PATH, transform=transform)
    print(f"数据集加载成功，共 {len(full_dataset)} 张图片。")

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # 损失函数（用于训练）和逐样本损失函数（用于筛选难样本）
    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f'开始第 {fold+1} 折训练...')
        
        # 初始化日志文件
        log_file = open(f"fold_{fold+1}_metrics.txt", "w", encoding="utf-8")
        log_file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_ids))
        val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_ids))

        model = UltimateSTFTClassifier(num_classes=len(full_dataset.classes)).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        early_stopping = EarlyStopping(patience=PATIENCE, path=f"model_fold_{fold+1}.pth")
        
        fold_hard_samples = [] # 存储该折所有验证样本的 (Loss, Path)

        for epoch in range(EPOCHS):
            # --- Training ---
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for imgs, labels, _ in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * imgs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss / train_total
            avg_train_acc = 100 * train_correct / train_total

            # --- Validation ---
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            current_epoch_samples = []

            with torch.no_grad():
                for imgs, labels, paths in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    
                    # 计算总 Loss 用于进度监控
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

                    # 如果是最后一个 Epoch 或者触发了早停，我们需要这里记录的 Loss 来选难样本
                    # 这里每一轮都记一下，或者只在最后一轮记
                    if early_stopping.counter == 0: # 只记录目前表现最好的一轮的样本情况
                        individual_losses = criterion_none(outputs, labels)
                        for i in range(len(paths)):
                            current_epoch_samples.append((individual_losses[i].item(), paths[i]))

            avg_val_loss = val_loss / val_total
            avg_val_acc = 100 * val_correct / val_total

            # 输出到控制台
            print(f"Epoch {epoch+1:02d}: Train Loss {avg_train_loss:.4f} Acc {avg_train_acc:.2f}% | "
                  f"Val Loss {avg_val_loss:.4f} Acc {avg_val_acc:.2f}%")
            
            # 写入日志文件
            log_file.write(f"{epoch+1},{avg_train_loss:.4f},{avg_train_acc:.2f},{avg_val_loss:.4f},{avg_val_acc:.2f}")

            # 早停与样本记录更新
            early_stopping(avg_val_loss, model)
            if early_stopping.counter == 0:
                fold_hard_samples = current_epoch_samples # 更新为当前表现最好时期的样本 Loss 情况
            
            if early_stopping.early_stop:
                print(f"--- 第 {fold+1} 折触发早停 ---")
                break
        
        log_file.close()

        # --- 筛选并保存难样本 ---
        # 按 Loss 降序排列
        fold_hard_samples.sort(key=lambda x: x[0], reverse=True)
        num_hard = int(len(fold_hard_samples) * HARD_SAMPLE_RATIO)
        top_hard_samples = fold_hard_samples[:num_hard]

        with open(f"fold_{fold+1}_hard_samples.txt", "w", encoding="utf-8") as f:
            f.write(f"Fold {fold+1} Hard Samples (Top {HARD_SAMPLE_RATIO*100}% Loss)\n")
            f.write("Loss, FilePath\n")
            for loss_val, path in top_hard_samples:
                f.write(f"{loss_val:.6f}, {path}\n")
        
        print(f"第 {fold+1} 折完成。难样本列表已保存。")

if __name__ == "__main__":
    main()
