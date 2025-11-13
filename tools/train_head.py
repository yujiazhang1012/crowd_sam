import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import random
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 防止在无GUI环境报错
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append("/home/ccnu-train/zyj/crowd_sam/crowd_sam")

from crowdsam.model import CrowdSAM
from crowdsam.data import CrowdHuman, collate_fn_crowdhuman
import crowdsam.utils as utils


def get_train_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class AugmentedCrowdHuman(CrowdHuman):
    def __init__(self, dataset_root, annot_path, use_sam_masks=True, class_counts=None, is_train=True):
        super().__init__(dataset_root, annot_path, transform=None, use_sam_masks=use_sam_masks)
        self.class_counts = class_counts or [520, 920, 4045, 915, 569, 58, 6]
        self.rare_classes = {i for i, count in enumerate(self.class_counts) if count < 200}
        self.train_transform = get_train_transform()
        self.val_transform = get_val_transform()
        self.is_train = is_train  # 控制是否使用增强

    def __getitem__(self, idx):
        roi, label, img_info = super().__getitem__(idx)
        
        if isinstance(roi, torch.Tensor):
            roi = T.ToPILImage()(roi)
        elif isinstance(roi, np.ndarray):
            roi = Image.fromarray(roi.astype('uint8'), 'RGB')
        # else:
        #     roi = Image.open(roi).convert('RGB')

        elif not isinstance(roi, Image.Image):
            raise TypeError(f"Unexpected roi type: {type(roi)}")
        # 如果已经是 Image.Image，直接跳过转换

        roi = roi.convert('RGB')  # 确保三通道

        # 验证模式：直接使用 val_transform
        if not self.is_train:
            roi = self.val_transform(roi)
            return roi, label, img_info

        # 训练模式：使用增强
        if label in self.rare_classes or label in [0,1,3]:
            roi = self._strong_augment(roi)
        else:
            if random.random() < 0.3:
                roi = self.train_transform.transforms[0](roi)
                roi = self.train_transform.transforms[1](roi)
                if random.random() < 0.5:
                    roi = T.RandomHorizontalFlip(p=1.0)(roi)
                if random.random() < 0.3:
                    roi = T.ColorJitter(brightness=0.1, contrast=0.1)(roi)
                roi = T.ToTensor()(roi)
                roi = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(roi)
            else:
                roi = self.val_transform(roi)
        # 获取原始图像尺寸
        # h_img = img_info.get('height',224), 
        # w_img = img_info.get('width',224)
        return roi, label, img_info

    def _strong_augment(self, img):

        # 对小样本类别使用更强的增强
        img = T.Resize((256, 256))(img)
        img = T.RandomCrop(224)(img)
        if random.random() < 0.5:
            img = T.RandomHorizontalFlip(p=0.5)(img)
        img = T.RandomRotation(degrees=20)(img)
        img = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15)(img)
    
        img = T.ToTensor()(img)
        img = T.RandomErasing(p=0.4, scale=(0.02, 0.15))(img)  
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.weight = weight

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def validate_model(model, val_dataloader, device, class_names, result_dir, epoch):
    model.eval()
    all_preds = []
    all_labels = []

    

    with torch.no_grad():
        for rois, labels, _ in tqdm(val_dataloader, desc="Validating"):
            rois = rois.to(device)
            labels = labels.to(device)
            features = model.action_backbone(rois)
            logits = model.action_head(features)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n Validation Accuracy: {acc:.4f}")
    print("\n Classification Report:")
    labels = list(range(len(class_names)))  # [0,1,2,3,4,5,6]
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        labels=labels,      # 👈 关键修复
        digits=4, 
        zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    cm_path = os.path.join(result_dir, f'confusion_matrix_epoch{epoch+1:02d}.png')    
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to:{cm_path}")
    plt.close()
    model.train()
    return acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', default='configs/crowdhuman.yaml')
    args = parser.parse_args()
    config = utils.load_config(args.config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 混淆矩阵保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_parent_dir = "results"               
    result_dir = os.path.join(results_parent_dir, f"results_{timestamp}")  
    os.makedirs(result_dir, exist_ok=True)
    print(f" Training logs and confusion matrices will be saved to: {result_dir}")

    # 初始化模型
    model = CrowdSAM(config, logger=None)
    model.to(model.device)

    for param in model.parameters():
        param.requires_grad = False
    # 解冻 EfficientNet 最后两个 block（features.7 和 features.8）
    for name, param in model.action_backbone.named_parameters():
        if "features.7" in name or "features.8" in name:
            param.requires_grad = True
            print(f"Unfrozen: {name}")
    for param in model.action_head.parameters():
        param.requires_grad = True
    # 为不同的类别分配不同的权重
    class_counts = torch.tensor([520, 920, 4045, 915, 569, 58, 6], dtype=torch.float).to(device)
    beta=0.9999
    effective_num=1.0 - torch.pow(beta, class_counts)
    class_weights = (1.0 - beta)/ effective_num
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # 归一化
    print(f"[INFO] Class Weights: {class_weights.tolist()}")
    # 优化器分层设置学习率
    optimizer = optim.AdamW([
        {'params': model.action_head.parameters(), 'lr': 1e-5},  # 分类头学习率高
        {'params': [p for n, p in model.action_backbone.named_parameters() if "7" in n or "8" in n], 'lr': 1e-5}  # backbone学习率低
    ], weight_decay=1e-5)

    class_counts_list = [520, 920, 4045, 915, 569, 58, 6]
    train_dataset = AugmentedCrowdHuman(
        dataset_root=config['data']['dataset_root'],
        annot_path=config['data']['train_file'],
        use_sam_masks=True,
        class_counts=class_counts_list,
        is_train=True
    )
    # 构建平衡验证层
    all_indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(
        all_indices, 
        test_size=0.1, 
        stratify=[train_dataset[i][1] for i in all_indices],
        random_state=42)
    val_dataset = AugmentedCrowdHuman(
        dataset_root=config['data']['dataset_root'],
        annot_path=config['data']['json_file'],
        use_sam_masks=True,
        class_counts=class_counts_list,
        is_train=False  
    )
    # 创建训练集和验证集子集
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn_crowdhuman)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn_crowdhuman)

    criterion = LabelSmoothingCrossEntropy(weight = class_weights, smoothing = 0.1).to(device)

    best_acc = 0.0
    for epoch in range(30):
        model.train()
        total_loss = 0
        total_samples = 0
        valid_samples = 0
        for rois, labels, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            rois = rois.to(model.device)
            labels = labels.to(model.device)
            optimizer.zero_grad()
            features = model.action_backbone(rois)
            cnn_features_flat = F.adaptive_avg_pool2d(features, 1).flatten(1)

           
            logits = model.action_head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == 29:
            val_acc = validate_model(
                model, 
                val_dataloader, 
                model.device, 
                class_names=["write", "read", "lookup", "turn_head", "raise_hand", "stand", "discuss"],
                result_dir = result_dir,
                epoch = epoch
            )
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs("weights", exist_ok=True)
                torch.save(model.action_head.state_dict(), "weights/action_head_best.pth")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.action_head.state_dict(), "weights/action_head_final.pth")
    print("Training completed!")